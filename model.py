import torch
import gym
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch import optim
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.tune.tune import run_experiments
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.annotations import override, DeveloperAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import one_hot
from ray.rllib.utils.typing import ModelConfigDict, TensorType
run_experiments()

# Encoder in Seq2Seq
class EncoderRNN(nn.Module):
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
class Attn(nn.Module):  
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
class AttnDecoderRNN(nn.Module):
     ## output_size 对应输出size，对应one-hot维度
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        
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


class Seq2Seq2Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> (TensorType, List[TensorType]):
        return super().forward(input_dict, state, seq_lens)    