import torch
import time
import numpy as np
from road_net import flow_params
from flow.utils.registry import make_create_env


# 参考 Experiment.run 源代码
def train(env, encoder, decoder, encoder_optim, decoder_optim) -> int:
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
    # information
    total_reward = 0
    vel = []
    state = env.reset()
    '''
    训练过程
    '''
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(torch.LongTensor(state), encoder_hidden)
    print(f'Out: {encoder_outputs}')
    decoder_input = Variable(torch.LongTensor([0, 0, 0, 0]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden
    for i in range(num_steps):
        action, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        t0 = time.time()
        state, reward, done, _ = env.step(action)
        t1 = time.time()
        times.append(1 / (t1 - t0))
        # Compute the velocity speeds and cumulative returns.
        veh_ids = env.k.vehicle.get_ids()
        vel.append(np.mean(env.k.vehicle.get_speed(veh_ids)))
        # record the reward and take the action as the next decoder input
        total_reward += reward
        decoder_input = Variable(torch.LongTensor(action))
        # 结束
        if done:
            break
    # Store the information from the run in info_dict.
    outflow = env.k.vehicle.get_outflow_rate(int(500))
    info_dict["returns"].append(total_reward)
    info_dict["velocities"].append(np.mean(vel))
    info_dict["outflows"].append(outflow)
    # Save emission data at the end of every rollout. This is skipped
    # by the internal method if no emission path was specified.
    if env.simulator == "traci":
        env.k.simulation.save_emission(run_id=0)
    # Print the averages/std for all variables in the info_dict.
    for key in info_dict.keys():
        print("Average, std {}: {}, {}".format(
            key, np.mean(info_dict[key]), np.std(info_dict[key])))
    print("Total time:", time.time() - t)
    print("steps/second:", np.mean(times))
    env.terminate()
    return total_reward


if __name__ == "__main__":
    # 首先我们需要根据road_net中定义的路网参数创造出对应的Enviroment
    create_env , _ = make_create_env(flow_params)
    env = create_env()
    # Seq2Seq Network
    encoder = EncoderRNN(156, 64)
    decoder = AttnDecoderRNN(attn_model='general', hidden_size=64, output_size=4)
    # optim
    encoder_optim = optim.SGD(encoder.parameters(), lr=0.001)
    decoder_optim = optim.SGD(decoder.parameters(), lr=0.001)

    num_epochs = 5
    for i in range(num_epochs):
        total_reward = train(env, encoder, decoder, encoder_optim, decoder_optim)
        # calculate loss
        # optim parameters
