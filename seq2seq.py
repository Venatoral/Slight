import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from road_net import flow_params
from flow.utils.registry import make_create_env


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
        
    def forward(self, word_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1) ## seq_len * 1 * hidden_size
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
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()  ## output_size 对应译文语言单词数，对应one-hot  维度
        
        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
                                              ##将上一次预测的词语和attention结果（encoder output）加权  拼起来
        self.out = nn.Linear(hidden_size * 2, output_size)
                                         ## 通过fc将结果转变成one-hot变量 
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
    
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x 1 *  hidden_size
        
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)  ## 拼凑起来
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs) ## 1 * 1 * seq_len
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # attention result, details in next line
                        # 1 * 1 * seq_len  mul  1 * seq_len * hidden_size =>  1 * 1 * hidden_size
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # size: 1 * hidden_size
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        #  1 * (2*hidden_size)  ->  1 * output_size
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights


# 参考 Experiment.run 源代码
def train(params=None, num_runs=1):
    # 首先我们需要根据road_net中定义的路网参数创造出对应的Enviroment
    create_env , _ = make_create_env(params=flow_params)
    env = create_env()
    # 接着与环境交互，结合seq2seq
    num_steps = env.env_params.horizon
    # used to store
    info_dict = {
        "returns": [],
        "velocities": [],
        "outflows": [],
    }
    # time profiling information
    t = time.time()
    times = []
    '''
    训练过程
    '''
    encoder = EncoderRNN()
    decoder = AttnDecoderRNN()
    def gen_action(*_):
        return None
    for i in range(num_runs):
        ret = 0
        vel = []
        state = env.reset()
        for j in range(num_steps):
            t0 = time.time()
            '''
            这一步比较关键，step中的参数应当设置成将state输入到我们的网络对应生成的action
            def gen_action(state):
                输入state进入我们的网络
                ...
                return clip_actions(out)
            '''
            state, reward, done, _ = env.step(gen_action(state))
            t1 = time.time()
            times.append(1 / (t1 - t0))
            # Compute the velocity speeds and cumulative returns.
            veh_ids = env.k.vehicle.get_ids()
            vel.append(np.mean(env.k.vehicle.get_speed(veh_ids)))
            ret += reward
            # 结束
            if done:
                break
        # Store the information from the run in info_dict.
        outflow = env.k.vehicle.get_outflow_rate(int(500))
        info_dict["returns"].append(ret)
        info_dict["velocities"].append(np.mean(vel))
        info_dict["outflows"].append(outflow)
        print("Round {0}, return: {1}".format(i, ret))
        # Save emission data at the end of every rollout. This is skipped
        # by the internal method if no emission path was specified.
        if env.simulator == "traci":
            env.k.simulation.save_emission(run_id=i)
    # Print the averages/std for all variables in the info_dict.
    for key in info_dict.keys():
        print("Average, std {}: {}, {}".format(
            key, np.mean(info_dict[key]), np.std(info_dict[key])))
    print("Total time:", time.time() - t)
    print("steps/second:", np.mean(times))
    env.terminate()

    # """Run the given network for a set number of runs.

    # Parameters
    # ----------
    # num_runs : int
    #     number of runs the experiment should perform
    # rl_actions : method, optional
    #     maps states to actions to be performed by the RL agents (if
    #     there are any)
    # convert_to_csv : bool
    #     Specifies whether to convert the emission file created by sumo
    #     into a csv file

    # Returns
    # -------
    # info_dict : dict < str, Any >
    #     contains returns, average speed per step
    # """
    # create_env, _ = make_create_env(flow_params)
    # env = create_env()
    # num_steps = env.env_params.horizon

    # # used to store
    # info_dict = {
    #     "returns": [],
    #     "velocities": [],
    #     "outflows": [],
    # }

    # def rl_actions(*_):
    #     return None

    # # time profiling information
    # t = time.time()
    # times = []

    # for i in range(num_runs):
    #     ret = 0
    #     vel = []
    #     state = env.reset()
    #     for j in range(num_steps):
    #         t0 = time.time()
    #         state, reward, done, _ = env.step(rl_actions(state))
    #         t1 = time.time()
    #         times.append(1 / (t1 - t0))

    #         # Compute the velocity speeds and cumulative returns.
    #         veh_ids = env.k.vehicle.get_ids()
    #         vel.append(np.mean(env.k.vehicle.get_speed(veh_ids)))
    #         ret += reward

    #         if done:
    #             break

    #     # Store the information from the run in info_dict.
    #     outflow = env.k.vehicle.get_outflow_rate(int(500))
    #     info_dict["returns"].append(ret)
    #     info_dict["velocities"].append(np.mean(vel))
    #     info_dict["outflows"].append(outflow)

    #     print("Round {0}, return: {1}".format(i, ret))

    #     # Save emission data at the end of every rollout. This is skipped
    #     # by the internal method if no emission path was specified.
    #     if env.simulator == "traci":
    #         env.k.simulation.save_emission(run_id=i)

    # # Print the averages/std for all variables in the info_dict.
    # for key in info_dict.keys():
    #     print("Average, std {}: {}, {}".format(
    #         key, np.mean(info_dict[key]), np.std(info_dict[key])))

    # print("Total time:", time.time() - t)
    # print("steps/second:", np.mean(times))
    # env.terminate()

    # return info_dict

if __name__ == "__main__":
    train(params=flow_params, num_runs=1)
