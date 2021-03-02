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
# Encoder in Seq2Seq
class ExampleEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1): ## input_size指one-hot embedding之后的维度
        super(EncoderRNN, self).__init__()  ## hidden_size指的是RNN中使用的hidden state维度
                                 ## n_layers 使用RNN（GRU）层数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)  ## one hot ——> embedding  one-hot维度过大
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers) 
    def forward(self, inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(inputs)
        embedded = self.embedding(inputs).view(seq_len, 1, -1) ## seq_len * 1 * hidden_size
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size)) ## inital GRU hidden state
        if torch.cuda.is_available(): hidden = hidden.cuda()
        return hidden


 ## attention 机制
class ExampleAttn(nn.Module):  
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        
        self.method = method
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs): ## 1 * hidden_size, seq_len * 1 * hidden_size 
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len))  # shape:seq_len
        if torch.cuda.is_available(): attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)  # Shape:  seq_len -> 1 * 1 * seq_len
    
    def score(self, hidden, encoder_output):  ## 1 * hidden_size, 1 * hidden_size
        hidden = hidden.squeeze(0)   # 降维
        encoder_output = encoder_output.squeeze(0)
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


# Decoder
class ExampleAttnDecoderRNN(nn.Module):
     ## output_size 对应输出size，对应one-hot维度
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(Decoder, self).__init__()
        
        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        # 将上一次预测和attention结果（encoder output）加权  拼起来
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        # 通过fc将结果转变成one-hot变量                              
        self.out = nn.Linear(hidden_size * 2, output_size)                         
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
    
    def forward(self, input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input (last output)
        embedded = self.embedding(input).view(1, 1, -1) # S=1 x 1 *  hidden_size
        
        # Combine embedded input and last context, run through RNN
        rnn_input = torch.cat((embedded, last_context.unsqueeze(0)), 2)  ## 拼凑起来
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs) ## 1 * 1 * seq_len
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # attention result, details in next line
                        # 1 * 1 * seq_len  mul  1 * seq_len * hidden_size =>  1 * 1 * hidden_size
        # Final output layer (next prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # size: 1 * hidden_size
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        #  1 * (2*hidden_size)  ->  1 * output_size
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights

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