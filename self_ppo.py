import torch
import gym
from torch.distributions.transforms import StackTransform
import torch.nn as nn
import torch.nn.functional as F
from flow.utils.registry import make_create_env
from road_net import ROAD_PARAMS, flow_params
from torch.distributions import Categorical
from matplotlib import pyplot as plt


MAX_LENGTH=512
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


class EncoderRNNAtt(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNNAtt, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        '''
        input: (seq_len, batch, input_size)
        hidden: (num_layers * num_directions, batch, hidden_size)
        num_directions = 1
        '''
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # input: (batch, state_size) -> embedded: (batch, hidden_size)
        embedded = self.embedding(input).to(device)
        embedded = embedded.unsqueeze(0)
        # output:  (seq_len, batch, num_directions * hidden_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden


class DecoderRNNAtt(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(DecoderRNNAtt, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Linear(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size,
                          dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).to(device)
        embedded = self.dropout(embedded)
        embedded = embedded.unsqueeze(0)
        # embedded: (batch, hidden_size) -> (seq_len, batch, hidden_size)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), dim=1)),
            dim=1
        )
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights


class AttentionSeqModel(nn.Module):

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space):
        super(AttentionSeqModel, self).__init__()
        nn.Module.__init__(self)
        self.obs_size = obs_space.shape[1]
        self.num_light = obs_space.shape[0]
        self.action_size = action_space.n
        # 此处设置 Encoder 和 Decoder 的 hidden_size
        self.hidden_size = 128
        self.encoder = EncoderRNNAtt(self.obs_size, self.hidden_size)
        self.decoder = DecoderRNNAtt(
            self.action_size, self.hidden_size, max_length=self.num_light)

    def forward(self, obs):
        # encoder
        if len(obs.shape) < 3:
            obs = obs.unsqueeze(0)
        batch_size = obs.shape[0]
        encoder_outputs = torch.zeros(
            self.decoder.max_length, self.encoder.hidden_size).to(device)
        encoder_hidden = torch.zeros(
            (1, batch_size, self.hidden_size)).to(device)
        # 将路口状态一个个输入encoder得到最终的hidden作为Context
        for i in range(self.num_light):
            encoder_output, encoder_hidden = self.encoder(
                obs[:, i, :], encoder_hidden)
            encoder_outputs[i] = encoder_output[0, 0]
        # 直接将encoder的隐藏层作为decoder的隐藏层
        decoder_hidden = encoder_hidden
        decoder_input = torch.zeros(
            size=(batch_size, self.action_size)).to(device)
        for i in range(self.num_light):
            logits, decoder_hidden, decoder_attn = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # record the actions of this intersection
            decoder_input = logits

        return logits


class ActorCritic(nn.Module):
    def __init__(self, env: gym.Env, hidden_size=128) -> None:
        super(ActorCritic, self).__init__()
        # actor to get action prob
        '''
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1),
        )
        '''
        obs_space = env.observation_space
        action_space = env.action_space
        obs_dim = obs_space.shape[0] * obs_space.shape[1]
        self.actor = AttentionSeqModel(obs_space, action_space)
        # critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def act(self, obs, memory):
        state = torch.from_numpy(obs).float().to(device)
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

        if len(state.shape) != 2:
            state = state.reshape((state.shape[0], -1))
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO():
    def __init__(self, env: gym.Env, gamma=0.99, eps_clip=0.2, lr=0.002, K_epochs=4, betas=(0.9, 0.999)) -> None:
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(env)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas
        )
        self.policy_old = ActorCritic(env)
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
        old_states = torch.stack(memory.states).detach().to(device)
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.probs).detach()

        # update network for K_epochs time
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)
            # new / old
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # update
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        # copy old network to new one
        self.policy_old.load_state_dict(self.policy.state_dict())


if __name__ == '__main__':
    # 首先注册环境
    create_env, env_name = make_create_env(flow_params)
    env = create_env()
    # PPO agent
    agent = PPO(env)
    rewards = []
    num_epochs = 2000
    batch_size = 2000
    timestep = 0
    memory = ExperiencePool()
    for epi in range(num_epochs):
        state = env.reset()
        total_reward = 0
        print('Epochs {} start...'.format(epi))
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
