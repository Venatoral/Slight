import torch
import torch.nn as nn
import math
import gym
import numpy as np
from typing import Dict, List, Tuple
from torch import device, nn
from torch.nn import functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch.nn.utils import weight_norm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, key_size, value_size, stride, dilation, padding, attn_heads=3, dropout=0.2, mask=None):
        super(TemporalBlock, self).__init__()
        # ConvNet1
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)
        # ConvNet2
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)

        self.convNet1 = nn.Sequential(
            self.conv1,
          #  self.chomp1,
            self.tanh1,
            self.dropout1,
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.tanh = nn.Tanh()
        # Attention
        self.attn_heads = attn_heads
        if attn_heads > 1:
            self.attn = [AttentionBlock(n_inputs, key_size, value_size, mask) for _ in range(attn_heads)]
            for i, attention in enumerate(self.attn):
                self.add_module('attention_{}'.format(i), attention)
            self.linear_cat = nn.Linear(n_inputs * attn_heads, n_inputs)
        else:
            self.attn = AttentionBlock(n_inputs, key_size, value_size, mask)
        self.use_attn = True
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        en_res_x = None
        if self.use_attn:
            # multi-head attention
            if self.attn_heads > 1:
                x_attn = []
                attn_weights = []
                for att in self.attn:
                    attn_out = att(x)
                    x_attn.append(attn_out[0])
                    attn_weights.append(attn_out[1])

                x_attn = torch.cat(x_attn, dim=1).transpose(1, 2)
                attn_weights = torch.stack(attn_weights).mean(dim=0)
                x_attn = self.linear_cat(x_attn).transpose(1, 2)
            # single attention
            else:
                x_attn, attn_weights = self.attn(x)


            weight_x = F.softmax(attn_weights.sum(dim=2), dim=1)
            en_res_x = weight_x.unsqueeze(2).repeat(1, 1, x.size(1)).transpose(1, 2) * x
            en_res_x = en_res_x if self.downsample is None else self.downsample(en_res_x)

            out = self.convNet1(x_attn)
        else:
            out = self.convNet1(x)

        res = x if self.downsample is None else self.downsample(x)

        if en_res_x is None:
            return self.tanh(out + res)
        else:
            return self.tanh(out + res + en_res_x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size, mask=None):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)
        self.mask = torch.ByteTensor(np.array(mask)).to(device)


    def forward(self, inputs):        
        inputs = inputs.permute(0,2,1) # inputs: [N, T, inchannels]
        keys = self.linear_keys(inputs) # keys: (N, T, key_size)
        query = self.linear_query(inputs) # query: (N, T, key_size)
        values = self.linear_values(inputs) # values: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        if self.mask is not None:
            temp.data.masked_fill_(self.mask, -float('inf'))
        # temp: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        weight_temp = F.softmax(temp / self.sqrt_key_size, dim=1) 
        value_attentioned = torch.bmm(weight_temp, values).permute(0,2,1) # shape: (N, T, value_size)
        return value_attentioned, weight_temp # value_attentioned: [N, in_channels, T], weight_temp: [N, T, T]


class TCNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        super(TCNModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        print('__init__: TCNModel')
        self.num_tl = obs_space.shape[0]
        self.input_size = obs_space.shape[1]
        self.output_size = num_outputs // self.num_tl
        # regulation
        self.I4R = "MaxDivideMin"
        self.reg_coef = 0.01
        self.hiddens = None
        # set TCN network
        layers = []
        
        num_channels = [16, 32, self.output_size]
        kernel_size = model_config['custom_model_config']['kernel_size']
        mask = model_config['custom_model_config']['adj_mask']

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channel = self.input_size if i == 0 else num_channels[i - 1]
            out_channel = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channel, out_channel, kernel_size,
                    key_size=self.num_tl, value_size=in_channel,
                    stride=1, dilation=dilation_size, padding=dilation_size, mask=mask)
                )
        # TCN
        self.net = nn.Sequential(*layers)
        # Critic
        self.ciritc = nn.Sequential(
            nn.Linear(self.input_size * self.num_tl, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )



    def forward(self, input_dict, state, seq_lens):
        self._value_out = self.ciritc(input_dict['obs_flat'])
        obs = input_dict['obs'].transpose(1, 2)
        prev = obs
        for i in range(len(self.net)):
            out = self.net[i](prev)
            prev = out
            if i == len(self.net) - 2:
                self.hiddens = out

        out = out.transpose(1, 2)
        out = out.reshape(shape=(out.shape[0], - 1))
        return out, state


    def value_function(self):
        assert self._value_out is not None, "must call forward() first"
        return self._value_out.squeeze(dim=1)
