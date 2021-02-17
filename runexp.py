import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv
from ray.rllib.examples.env.repeat_initial_obs_env import RepeatInitialObsEnv
from ray.rllib.examples.models.rnn_model import TorchRNNModel
from model import Seq2Seq2Model
from ray.rllib.models import ModelCatalog
from flow.utils.registry import make_create_env
from road_net import ROAD_PARAMS, flow_params

# 路口数量
INTER_NUM = ROAD_PARAMS['n_rows'] * ROAD_PARAMS['n_columns']

if __name__ == "__main__":
    # 首先注册环境
    create_env , gym_name = make_create_env(flow_params)
    env = create_env()
    register_env(gym_name, create_env)
    # ray 集群环境
    ray.init(num_cpus=4)
    ModelCatalog.register_custom_model(
        "seq2seq", Seq2Seq2Model
        )
    # register_env("RepeatAfterMeEnv", lambda c: RepeatAfterMeEnv(c))
    # register_env("RepeatInitialObsEnv", lambda _: RepeatInitialObsEnv())

    config = {
        "env": gym_name,
        "env_config": {
            "repeat_delay": 2,
        },
        "gamma": 0.9,
        "entropy_coeff": 0.001,
        "num_sgd_iter": 5,
        "vf_loss_coeff": 1e-5,
        "model": {
            "custom_model": "seq2seq",
            "custom_model_config":{
                'inter_num': INTER_NUM
            }
        },
        "framework": "torch",
    }
    print(config['model']['custom_model'])
    results = tune.run('PPO', config=config)
    ray.shutdown()
