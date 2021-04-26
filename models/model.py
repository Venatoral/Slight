import torch
import gym
from typing import Dict, List, Tuple
from torch import device, nn
from torch.nn import functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from .rugularization import MaxDivideMin
from .rugularization import MaxMinusMin

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 512

class EncoderRNNAtt(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNNAtt, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        '''
        input: (seq_len, batch, input_size)
        hidden: (num_layers * num_directions, batch, hidden_size)
        num_directions = 1
        '''
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # input: (batch, state_size) -> embedded: (batch, hidden_size)
        embedded = self.embedding(input).to(device)
        embedded = embedded.unsqueeze(0)
        # output:  (seq_len, batch, num_directions * hidden_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden, embedded[0][0]


class DecoderRNNAtt(nn.Module):

    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(DecoderRNNAtt, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        # layers
        self.embedding = nn.Linear(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, hidden, encoder_outputs, attn_mask, use_attn=True):
        embedded = self.embedding(input).to(device)
        embedded = self.dropout(embedded)
        # embedded: (batch, hidden_size) -> (seq_len, batch, hidden_size)
        embedded = embedded.unsqueeze(0)
        # use attention
        if use_attn:
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded[0], hidden[0]), dim=1)),
                dim=1
            )
            # use attention mask to concern specialy abount neightbors
            attn_weights = (attn_weights * attn_mask)
            attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                     encoder_outputs).squeeze(1)
            output = torch.cat((embedded[0], attn_applied), 1)
            output = self.attn_combine(output).unsqueeze(0)
        else:
            output = self.fc(embedded)
        output = torch.tanh(output)
        output, hidden = self.gru(output, hidden)

        # output = F.log_softmax(self.out(output[0]), dim=1)
        output = torch.tanh(self.out(output[0]))
        return output, hidden, attn_weights


class AttentionSeqModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        print('__init__: AttentionModel')
        super(AttentionSeqModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # the adj matrix for attention usage
        self.attention_mask = torch.nn.Parameter(
            torch.from_numpy(model_config['custom_model_config']['adj_mask']).float(),
            requires_grad=True,
        )
        # dimensions
        self.obs_size = obs_space.shape[1]
        self.num_light = obs_space.shape[0]
        self.action_size = 2
        self.name = name
        # 此处设置 Encoder 和 Decoder 的 hidden_size
        self.hidden_size = 128
        self.encoder = EncoderRNNAtt(self.obs_size, self.hidden_size)
        self.decoder = DecoderRNNAtt(self.action_size, self.hidden_size, max_length=self.num_light, dropout_p=0)
        # value function
        self._value_branch = nn.Sequential(
            nn.Linear(self.obs_size * self.num_light, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.time_step = 0
        self._value_out = None
        self.hiddens=None
        self.I4R="MaxMinusMin"
        self.reg_coef=0.01
        #or "MaxMinusMin"

    '''
    input_dict (dict) – dictionary of input tensors, 
    including “obs”, “obs_flat”, “prev_action”, “prev_reward”,
     “is_training”, “eps_id”, “agent_id”, “infos”, and “t”.

    @return [BATCH, num_outputs], and the new RNN state.
    '''
    def custom_loss(self, policy_loss: TensorType,
                    loss_inputs: Dict[str, TensorType]) -> TensorType:
        """Override to customize the loss function used to optimize this model.

        This can be used to incorporate self-supervised losses (by defining
        a loss over existing input and output tensors of this model), and
        supervised losses (by defining losses over a variable-sharing copy of
        this model's layers).

        You can find an runnable example in examples/custom_loss.py.

        Args:
            policy_loss (Union[List[Tensor],Tensor]): List of or single policy
                loss(es) from the policy.
            loss_inputs (dict): map of input placeholders for rollout data.

        Returns:
            Union[List[Tensor],Tensor]: List of or scalar tensor for the
                customized loss(es) for this model.
        """
        return policy_loss
        print("[2]hiddens final shape: ", self.hiddens.shape)

        if self.I4R == "MaxDivideMin":
            reguModel = MaxDivideMin.apply
            norm=reguModel(self.hiddens, self.reg_coef)
        elif self.I4R == "MaxMinusMin":
            reguModel = MaxMinusMin.apply
            norm=reguModel(self.hiddens, self.reg_coef)
        else:
            norm=0
        norm=norm.to(device)
        print("norm: ", norm)
        policy_loss[0]=policy_loss[0]+norm

        self.hiddens = None
        return policy_loss

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[
        TensorType, List[TensorType]]:
        # record time steps
        self.time_step += 1
        # obs
        obs: torch.Tensor = input_dict['obs'].float().to(device)
        # 记录value
        self._value_out = self._value_branch(input_dict['obs_flat'])
        # 记录 batch_size
        self._last_batch_size = obs.shape[0]
        # num of lights
        # encoder
        # hidden: (num_layers * num_directions, batch, hidden_size)
        encoder_outputs = torch.zeros(
            self._last_batch_size, self.decoder.max_length, self.encoder.hidden_size).to(device)
        encoder_hidden = torch.zeros(
            (1, self._last_batch_size, self.hidden_size)).to(device)
        hidden=None
        # 将路口状态一个个输入encoder得到最终的hidden作为Context
        for i in range(self.num_light):
            encoder_output, encoder_hidden, embedded = self.encoder(
                obs[:, i, :], encoder_hidden)
            encoder_outputs[:, i, :] = encoder_output[0, :, :]
            # if hidden is None:
            #     hidden=embedded.detach()
            #     hidden=hidden.unsqueeze(0)
            #     # print("hidden start shape: {}".format(hidden.shape))
            # else:
            #     embedded=embedded.unsqueeze(0)
            #     hidden=torch.cat((hidden, embedded.detach()), 1)
                # print("hidden after shape: {}".format(hidden.shape))
        # print("*" * 30)
        #hidden [1, 36*128]
    #     if self.hiddens is None:
    #         self.hiddens=hidden.detach()
    # #        print("[0]hiddens shape: ", self.hiddens.shape)
    #         # print("hiddens start shape: ", self.hiddens.shape)
    #     else:
    #         self.hiddens=torch.cat([self.hiddens, hidden.detach()])
    #        print("[1]hiddens shape: ", self.hiddens.shape)
            # print("hiddens after shape: ", self.hiddens.shape)
        #hiddens shape torch.Size([2, 4608])

        # 直接将encoder的隐藏层作为decoder的隐藏层
        decoder_hidden = encoder_hidden
        decoder_input = torch.zeros(
            size=(self._last_batch_size, self.action_size)).to(device)
        outs = torch.zeros(
            size=(self._last_batch_size, self.num_light, self.action_size)).to(device)
        # attn recorder
        # attns = []
        for i in range(self.num_light):
            decoder_out, decoder_hidden, decoder_attn = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs, self.attention_mask[i]
            )
            # record the actions of this intersection
            # attns.append(decoder_attn[0].detach().cpu().numpy().tolist())
            decoder_input = decoder_out.detach()
            outs[:, i, :] = decoder_out
        # draw matrix
        # if (self.time_step + 1) % 500 == 0:
        #     print('Save Attns!')
        #     df = pd.DataFrame(attns)
        #     try:
        #         df.to_csv('attention_{}.csv'.format(self.name))
        #     except:
        #         pass

        outs = outs.reshape(shape=(outs.shape[0], -1))
        return outs, state

    def value_function(self) -> TensorType:
        assert self._value_out is not None, "must call forward() first"
        return self._value_out.squeeze(dim=1)
