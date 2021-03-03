import torch
import gym
import numpy as np
from typing import Dict, List, Tuple
from torch import dtype, nn, tensor, transpose
from torch.autograd import Variable
from torch.nn import functional as F
# from ray.rllib.examples.attention_net
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.torch.misc import SlimFC, normc_initializer

MAX_LENGTH = 512
#imitate the pytorch attention tutorial
class EncoderRNNAtt(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNNAtt, self).__init__()
        self.hidden_size = hidden_size
        # (*) -> (*, embedding_dim)
        self.embedding = nn.Embedding(input_size, hidden_size)
        '''
        input: (seq_len, batch, input_size)
        hidden: (num_layers * num_directions, batch, hidden_size)
        num_directions = 1
        '''
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # input: (batch, seq_len) -> embedded: (batch, seq_len, hidden_size)
        input = input.long()
        embedded = self.embedding(input)
        # embedded: (batch, seq_len, hidden_size) -> (seq_len, batch, hidden_size)
        embedded = embedded.transpose(1, 0)
        # output:  (seq_len, batch, num_directions * hidden_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden


class DecoderRNNAtt(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(DecoderRNNAtt,self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input.long())
        embedded = self.dropout(embedded)
        # embedded: (batch, seq_len, hidden_size) -> (seq_len, batch, embedding_dim)
        embedded = embedded.transpose(1, 0)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

class AttentionModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        super(AttentionModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.obs_size = obs_space.shape[0]
        self.action_size = action_space.n
        self.num_outputs = num_outputs
        # 此处设置 Encoder 和 Decoder 的 hidden_size
        self.hidden_size = 128
        self.encoder = EncoderRNNAtt(self.obs_size, self.hidden_size)
        self.decoder = DecoderRNNAtt(self.action_size, self.hidden_size)
        # 路口数量
        self.inter_num = model_config['custom_model_config'].get('inter_num')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._features = None
        # value function
        self._value_branch = SlimFC(
            in_size=num_outputs,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)

    '''
    input_dict (dict) – dictionary of input tensors, 
    including “obs”, “obs_flat”, “prev_action”, “prev_reward”,
     “is_training”, “eps_id”, “agent_id”, “infos”, and “t”.
    
    @return [BATCH, num_outputs], and the new RNN state.
    '''
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        obs: torch.Tensor = input_dict['obs_flat'].float()
        self._last_batch_size = obs.shape[0]
        # encoder
        # hidden: (num_layers * num_directions, batch, hidden_size)
        encoder_outputs = torch.zeros(self.decoder.max_length, self.encoder.hidden_size)
        encoder_hidden = torch.zeros((1, self._last_batch_size, self.hidden_size))
        # obs -> (seq_len, batch_size, input_size)
        obs = obs.reshape((self.inter_num, self._last_batch_size, -1))
        for i in range(obs.shape[0]):
            encoder_output, encoder_hidden = self.encoder(obs[i], encoder_hidden)
            encoder_outputs[i] = encoder_output[0, 0]
        # 直接将encoder的隐藏层作为decoder的隐藏层 
        decoder_hidden = encoder_hidden
        outs = torch.zeros((self._last_batch_size, self.action_size))
        decoder_input = torch.zeros(size=(self._last_batch_size, self.action_size))
        for i in range(self._last_batch_size):
            decoder_out, decoder_hidden, decoder_attn = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            outs[i] = decoder_out[0]
            _, topi = decoder_out.topk(1)
            decoder_input = topi
        self._features = outs
        return outs, state

    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        value = self._value_branch(self._features).squeeze(1)
        return value