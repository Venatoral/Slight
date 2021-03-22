import ray
import os
import torch
from ray import tune
from ray.tune.registry import register_env
from models.model import AttentionSeqModel
from ray.rllib.models import ModelCatalog
from flow.utils.registry import make_create_env
from road_net import ROAD_PARAMS, flow_params
from matplotlib import pyplot as plt


# 路口数量
INTER_NUM = ROAD_PARAMS['n_rows'] * ROAD_PARAMS['n_columns']

if __name__ == "__main__":
    # 首先注册环境
    create_env , gym_name = make_create_env(flow_params)
    env = create_env()
    register_env(gym_name, create_env)
    # ray 集群环境
    cpu_num = os.cpu_count()
    gpu_num = 1 if torch.cuda.is_available() else 0
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
    # 修改 training_iteration 改变训练回合数
    results = tune.run(
        'PPO',
        num_samples=1,
        local_dir='./results', 
        config=config, 
        stop={"training_iteration": 1},
        resume=False,
        )
    
    ray.shutdown()
    print('Training is over!')
    print('Result is {}'.format(results))
