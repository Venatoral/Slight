import ray
import os
import torch
from ray import tune
from ray.tune.registry import register_env
from models.model import AttentionSeqModel
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTrainer
from flow.utils.registry import make_create_env
from road_net import ROAD_PARAMS, flow_params
from matplotlib import pyplot as plt
from ray.tune.logger import pretty_print




# 路口数量
INTER_NUM = ROAD_PARAMS['n_rows'] * ROAD_PARAMS['n_columns']


def train():
    # 首先注册环境
    create_env , gym_name = make_create_env(flow_params)
    env = create_env()
    register_env(gym_name, create_env)
    # ray 集群环境
    cpu_num = os.cpu_count()
    gpu_num =torch.cuda.device_count()
    ray.init(
        num_cpus=cpu_num,
        num_gpus=gpu_num
        )
    ModelCatalog.register_custom_model(
        "attentionModel", AttentionSeqModel,
        )
    # register_env("RepeatAfterMeEnv", lambda c: RepeatAfterMeEnv(c))
    # register_env("RepeatInitialObsEnv", lambda _: RepeatInitialObsEnv())

    config = {
        "env": gym_name,
        "env_config": {
            "repeat_delay": 2,
        },
        "gamma": 0.99,
        "entropy_coeff": 1e-3,
       # 'vf_clip_param': 200.0,
        "num_sgd_iter": 10,
        "vf_loss_coeff": 1e-5,
        'num_gpus': gpu_num,
        "model": {
            "custom_model": "attentionModel",
        },
        "framework": "torch",
    }
    # trainer = PPOTrainer(config)
    # # 迭代数
    # iteration_times = 500
    # # 保存 checkpoint 频率
    # save_freq = 10
    # for i in range(iteration_times):
    #     print('{} iteration begin...'.format(i))
    #     # Perform one iteration of training the policy with PPO
    #     result = trainer.train()
    #     print(pretty_print(result))

    #     if i % save_freq == 0:
    #         checkpoint = trainer.save()
    #         print("checkpoint saved at", checkpoint)
    # # 最后也保存一下checkpoint
    # trainer.save()
    # # 修改 training_iteration 改变训练回合数
    results = tune.run(
        'PPO',
        num_samples=1,
        local_dir='./results', 
        config=config,
        stop={"training_iteration": 1},
        resume=False,
        checkpoint_freq=5,
        checkpoint_at_end=True,
        )
    
    ray.shutdown()
    print('Training is over!')



def test(checkpoint_path):
    # 首先注册环境
    create_env , gym_name = make_create_env(flow_params)
    env = create_env()
    register_env(gym_name, create_env)
    # ray 集群环境
    cpu_num = os.cpu_count()
    gpu_num =torch.cuda.device_count()
    ray.init(
        num_cpus=cpu_num,
        num_gpus=gpu_num
        )
    ModelCatalog.register_custom_model(
        "attentionModel", AttentionSeqModel,
        )
    # register_env("RepeatAfterMeEnv", lambda c: RepeatAfterMeEnv(c))
    # register_env("RepeatInitialObsEnv", lambda _: RepeatInitialObsEnv())

    config = {
        "env": gym_name,
        "env_config": {
            "repeat_delay": 2,
        },
        "gamma": 0.99,
        "entropy_coeff": 1e-3,
       # 'vf_clip_param': 200.0,
        "num_sgd_iter": 10,
        "vf_loss_coeff": 1e-5,
        'num_gpus': gpu_num,
        "model": {
            "custom_model": "attentionModel",
        },
        "framework": "torch",
    }
    agent = PPOTrainer(config)
    #  加载 checkopoints
    agent.restore(checkpoint_path)
    num_epochs = 500
    rewards = []
    for epi in range(num_epochs):
        state = env.reset()
        total_reward = 0
        print('Epochs {} start...'.format(epi))
        while True:
            # 产生动作
            action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)
            # 记录r, done
            total_reward += reward
            if done:
                break
        print('{} epochs rewards: {}'.format(epi, total_reward))
        rewards.append(total_reward)
    plt.xlabel('epochs')
    plt.ylabel('reward')
    plt.plot(rewards)
    plt.savefig('{}_rewards.png'.format('PPO_Seq2Seq'))


if __name__ == "__main__":
    train()

