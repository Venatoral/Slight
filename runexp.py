from os import name
import ray
from ray import tune
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.trainer_template import build_trainer
from ray.tune.registry import register_env
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv
from ray.rllib.examples.env.repeat_initial_obs_env import RepeatInitialObsEnv
from ray.rllib.examples.models.rnn_model import TorchRNNModel
from ray.rllib.models import ModelCatalog
from flow.utils.registry import make_create_env
from ray.tune.tune import run
from road_net import flow_params

if __name__ == "__main__":
    # 首先注册环境
    create_env , gym_name = make_create_env(flow_params)
    env = create_env()
    register_env(gym_name, create_env)
    # ray 集群环境
    ray.init(num_cpus=4)
    ModelCatalog.register_custom_model(
        "rnn", TorchRNNModel
        )
    register_env("RepeatAfterMeEnv", lambda c: RepeatAfterMeEnv(c))
    register_env("RepeatInitialObsEnv", lambda _: RepeatInitialObsEnv())

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
            "custom_model": "rnn",
        },
        "framework": "torch",
    }
    
    results = tune.run('PPO', config=config)
    ray.shutdown()
