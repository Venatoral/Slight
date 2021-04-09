import torch
import gym
from typing import Dict, List, Tuple
from torch import device, nn
from torch.nn import functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class FullyConnectedModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        super(FullyConnectedModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.obs_size = obs_space.shape[0] * obs_space.shape[1]
        self.action_size = num_outputs
        self.hidden_size = 128
        # 此处设置 Encoder 和 Decoder 的 hidden_size
        self.embedding=nn.Linear(self.obs_size, self.hidden_size)
        self.dropout = nn.Dropout(0.1)

        # value function
        self._value_branch = nn.Sequential(
            nn.Linear(self.obs_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.lmodel=nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.action_size)
        )
        self._value_out = None

    '''
    input_dict (dict) – dictionary of input tensors, 
    including “obs”, “obs_flat”, “prev_action”, “prev_reward”,
     “is_training”, “eps_id”, “agent_id”, “infos”, and “t”.

    @return [BATCH, num_outputs], and the new RNN state.
    '''

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[
        TensorType, List[TensorType]]:

        obs: torch.Tensor = input_dict['obs_flat'].float().to(device)
        # 记录value
        self._value_out = self._value_branch(obs)
        # 记录 batch_size
        self._last_batch_size = obs.shape[0]

        embedded=F.tanh(self.embedding(obs).to(device))
        embedded=self.dropout(embedded)

        out = self.lmodel(embedded)
        return out, state

    def value_function(self) -> TensorType:
        assert self._value_out is not None, "must call forward() first"
        return self._value_out.squeeze(dim=1)