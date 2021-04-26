import ray
import os
import torch
from ray import tune
from ray.tune.registry import register_env
from models.model import AttentionSeqModel
from models.tcn import TCNModel
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.ppo import PPOTorchPolicy, DEFAULT_CONFIG
from ray.rllib.models import ModelCatalog
from flow.utils.registry import make_create_env
from road_net import ROAD_PARAMS, flow_params
from flow.core import rewards
import numpy as np


# 路口数量
INTER_NUM = ROAD_PARAMS['n_rows'] * ROAD_PARAMS['n_columns']


def train():
     # 首先注册环境
    create_env, gym_name = make_create_env(flow_params)
    env = create_env()
    register_env(gym_name, create_env)
    # ray 集群环境
    cpu_num = os.cpu_count()
    gpu_num = torch.cuda.device_count()
    ray.init(
        num_cpus=cpu_num,
        num_gpus=gpu_num
    )
    ModelCatalog.register_custom_model(
        "attnModel", AttentionSeqModel,
    )
    ModelCatalog.register_custom_model(
        "tcnModel", TCNModel,
    )
    # generate adj matrix
    adj_matrix = env.get_adj_matrix()
    config = {
        "env": gym_name,
        "env_config": {
            "repeat_delay": 1,
        },
        "gamma": 0.99,
        "entropy_coeff": 1e-3,
        'vf_clip_param': 20.0,
        "num_sgd_iter": 10,
        "vf_loss_coeff": 1e-5,
        'num_gpus': gpu_num,
        'num_workers':1,
        "model": {
            "custom_model": "tcnModel",
            "custom_model_config": {
                'kernel_size': 3
            }
        },
        "framework": "torch",
    }

    policy = PPOTorchPolicy
    trainer = build_trainer(name='PPOPlus', default_config=DEFAULT_CONFIG, default_policy=policy)
    # 修改 training_iteration 改变训练回合数
    results = tune.run(
        trainer,
        name='Desired',
        local_dir='./results', 
        config=config,
        stop={"training_iteration": 1},
        resume=False,
        # checkpoint_at_end=True,
        # checkpoint_freq=5,
        # keep_checkpoints_num=10,
        )
    ray.shutdown()
    print('Training is over!')
    print('Result is {}'.format(results))



def test(checkpoint_path: str, num_epochs=1):
     # 首先注册环境
    create_env, gym_name = make_create_env(flow_params)
    env = create_env()
    register_env(gym_name, create_env)
    # ray 集群环境
    cpu_num = os.cpu_count()
    gpu_num = torch.cuda.device_count()
    ray.init(
        num_cpus=cpu_num,
        num_gpus=gpu_num
    )
    ModelCatalog.register_custom_model(
        "attnModel", AttentionSeqModel,
    )
    # generate adj matrix
    adj_matrix = env.get_adj_matrix()
    config = {
        "env": gym_name,
        "env_config": {
            "repeat_delay": 2,
        },
        "gamma": 0.99,
        "entropy_coeff": 1e-3,
        'vf_clip_param': 20.0,
        "num_sgd_iter": 10,
        "vf_loss_coeff": 1e-5,
        'num_gpus': 0,
        "model": {
            "custom_model": "attnModel",
            "custom_model_config": {
                'adj_mask': adj_matrix
            }
        },
        "framework": "torch",
    }

    agent = PPOTrainer(config)
    agent.restore(checkpoint_path)
    # queue_len
    epoch_queue_len = []
    # aver_speed
    epoch_aver_speed = []
    # delay
    epoch_delay_time = []

    total_reward = []
    for epi in range(num_epochs):
        state = env.reset()
        # queue_len
        queue_len = []
        # aver_speed
        aver_speed = []
        # delay
        delay_time = []

        epi_rewards = 0
        print('Epochs {} start...'.format(epi))
        while True:
            # 产生动作
            action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)

            delay_time.append(rewards.min_delay_unscaled(env))
            aver_speed.append(rewards.average_velocity(env))
            queue_len.append(rewards.queue_length(env))
            epi_rewards += reward

            if done:
                break
        print('Episode {}:\nAverage Spped {}\nDelay Time:{}\nQueue Length:{}\nReward:{}'.format(
            epi,
            np.mean(aver_speed),
            np.mean(delay_time),
            np.mean(queue_len),
            epi_rewards,
        ))
    
        epoch_aver_speed.append(np.mean(aver_speed))
        epoch_delay_time.append(np.mean(delay_time))
        epoch_queue_len.append(np.mean(queue_len))
        total_reward.append(epi_rewards)

    print('Total:\nAverage Spped {}\nDelay Time:{}\nQueue Length:{}\nReward:{}'.format(
        np.mean(epoch_aver_speed),
        np.mean(epoch_delay_time),
        np.mean(epoch_queue_len),
        np.mean(total_reward)
    ))
    
    ray.shutdown()



if __name__ == "__main__":
    train()
   # test('/home/male/Desktop/Slight-dev/results/PPO/PPO_SeqTrafficLightEnv-v0_d606b_00000_0_2021-04-07_17-58-43/checkpoint_300/checkpoint-300')
