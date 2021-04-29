import torch
import torch.nn as nn
import gym
from typing import Dict, List, Tuple
from torch import device, nn
from torch.nn import functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
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
            self.chomp1,
            self.tanh1,
            self.dropout1,
        )
        
        self.convNet2 = nn.Sequential(
            self.conv2,
            self.chomp2,
            self.tanh2,
            self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.convNet1(x)
        # print('[1]. out.shape:{}'.format(out.shape))
        # out = self.convNet2(out)
        # print('[2]. out.shape:{}'.format(out.shape))
        res = x if self.downsample is None else self.downsample(x)
        return self.tanh(out + res)


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
        
        self.I4R = "MaxDivideMin"
        self.reg_coef = 0.01
        self.hiddens = None
        # set TCN network
        layers = []
        
        num_channels = [32, 64, self.output_size]
        kernel_size = model_config['custom_model_config']['kernel_size']
        
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channel = self.input_size if i == 0 else num_channels[i - 1]
            out_channel = num_channels[i]
            layers.append(
                TemporalBlock(
                    in_channel, out_channel, kernel_size, 
                    stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size)
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
        out = out.reshape(shape=(out.shape[0], - 1))
        return out, state


    def value_function(self):
        assert self._value_out is not None, "must call forward() first"
        return self._value_out.squeeze(dim=1)
