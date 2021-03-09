import torch
import torch.nn as nn
import torch.nn.functional as F
from flow.utils.registry import make_create_env
from road_net import ROAD_PARAMS, flow_params
from torch.distributions import Categorical
from matplotlib import pyplot as plt

class ExperiencePool():
    def __init__(self) -> None:
        self.actions = []
        self.probs = []
        self.states = []
        self.rewards = []
        self.dones = []

    def clear(self):
        del self.actions[:]
        del self.probs[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=64) -> None:
        # actor to get action prob
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
    def act(self, obs, memory):
        state = torch.from_numpy(obs).float()
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.probs.append(dist.log_prob(action))
        
        return action.item()

    def forward(self):
        raise NotImplementedError

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO():
    def __init__(self, obs_dim, action_dim, gamma=0.99, eps_clip=0.2, lr=0.002, K_epochs=4, betas=(0.9, 0.999)) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(obs_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(obs_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        rewards = []
        discounted_rewards = 0
        # get discounted rewards
        discounted_reward = 0
        for reward, done in zip(reversed(memory.rewards), reversed(memory.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # normalize rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # convert list to tensor
        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.probs).detach()

        # update network for K_epochs time
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # new / old
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            # update
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # copy old network to new one
        self.policy_old.load_state_dict(self.policy.state_dict())


if __name__ == '__main__':
    # 首先注册环境
    create_env , env_name = make_create_env(flow_params)
    env = create_env()
    # PPO agent
    agent = PPO(env.observation_space.shape[0], env.action_space.n)
    rewards = []
    num_epochs = 100
    batch_size = 2000
    timestep = 0
    memory = ExperiencePool()
    for epi in range(num_epochs):
        state = env.reset()
        total_reward = 0
        print('Epochs {} start...'.format(epi))
        print(state)
        while True:
            timestep += 1
            # 产生动作
            action = agent.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            # 记录r, done
            memory.rewards.append(reward)
            memory.dones.append(done)
            # batch 训练
            if timestep % batch_size == 0:
                agent.update(memory)
                memory.clear()
                timestep = 0
            # count rewards
            total_reward += reward
            if done:
                break
        print('{} epochs rewards: {}'.format(epi, total_reward))
        rewards.append(total_reward)
        if (epi + 1) % 5 == 0:
            torch.save(agent.policy.state_dict(), './PPO_{}'.format(env_name))
            print('model saved')
    plt.title('{} rewards curve'.format(env_name))
    plt.xlabel('epochs')
    plt.ylabel('reward')
    plt.plot(rewards)
    plt.savefig('{}_rewards.png'.format(env_name))