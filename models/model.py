from ray.rllib.examples.models.rnn_model import TorchRNNModel
import torch
import gym
import numpy as np
from typing import Dict, List, Tuple
from torch import dtype, nn, tensor
from torch.autograd import Variable
from torch.nn import functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.torch.misc import SlimFC, normc_initializer


# implement by ml
class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, embed_dim, layers) -> None:
        super(Encoder, self).__init__()
        # dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.layers = layers
        # gru and embeding
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=layers)
    
    def forward(self, src):
        src = src.long()
        embedded = self.embedding(src)
        out, hidden = self.gru(embedded)
        return out, hidden


class Decoder(nn.Module):
    
    def __init__(self, output_dim, hidden_dim, embed_dim, layers) -> None:
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.layers = layers
        # gru and embeding
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=layers)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        inputs = inputs.view(1, -1)
        # inputs = inputs.unsqueeze(0)
        embedded = nn.functional.relu(self.embedding(inputs))
        output, hidden = self.gru(embedded, hidden)
        predict = self.softmax(self.out(output[0]))
        return predict, hidden


class Seq2Seq2Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        super(Seq2Seq2Model, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.obs_size = obs_space.shape[0]
        self.action_size = action_space.n
        self.action_space = action_space
        self.num_outputs = num_outputs
        #demension 
        self.encoder = Encoder(self.obs_size, 128, 512, 1)
        self.decoder = Decoder(self.action_size, 128, 512, 1)
        # 路口数量
        self.inter_num = model_config.get('inter_num')
        # 使用GPU加速
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._features = None
        self._value_branch = SlimFC(
            in_size=num_outputs,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None)

    # input_dict (dict) – dictionary of input tensors
    # including “obs”, “obs_flat”, “prev_action”, 
    # “prev_reward”, “is_training”, 
    # “eps_id”, “agent_id”, “infos”, and “t”.
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict['obs_flat'].float().to(self.device)
        self._last_batch_size = obs.shape[0]
        print(f'batch_size: {self._last_batch_size}')
        # encoder
        # for i in range(obs.shape[0]):
        #     _, encoder_hidden = self.encoder(obs)
        # obs [32(batch_size), 156(obs_space.shape[0])]
        _, encoder_hidden = self.encoder(obs)
        # decoder (to fix, why 1, 156, 128 and expected 1, 1, 128)
        hidden = encoder_hidden[:, -1, :].unsqueeze(1)
        # to fix
        outs = torch.zeros((self._last_batch_size, self.action_size))
        decoder_input = torch.tensor([0])
        for i in range(self._last_batch_size):
            decoder_out, hidden = self.decoder(decoder_input, hidden)
            outs[i] = decoder_out[0]
            _, topi = decoder_out.topk(1)
            decoder_input = topi
        self._features = outs
        return outs, state


    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        value = self._value_branch(self._features).squeeze(1)
        return value