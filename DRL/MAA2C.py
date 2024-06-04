
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np

from common.Agent import Agent
from common.Model import ActorNetwork, CriticNetwork
from common.utils import entropy, index_to_one_hot, to_tensor_var

import matplotlib.pyplot as plt
import os
import time
import datetime
import pygame
import math

class MAA2C(Agent):
    """
    An multi-agent learned with Advantage Actor-Critic
    - Actor takes its local observations as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy

    Parameters
    - training_strategy:
        - cocurrent
            - each agent learns its own individual policy which is independent
            - multiple policies are optimized simultaneously
        - centralized (see MADDPG in [1] for details)
            - centralized training and decentralized execution
            - decentralized actor map it's local observations to action using individual policy
            - centralized critic takes both state and action from all agents as input, each actor
                has its own critic for estimating the value function, which allows each actor has
                different reward structure, e.g., cooperative, competitive, mixed task
    - actor_parameter_sharing:
        - True: all actors share a single policy which enables parameters and experiences sharing,
            this is mostly useful where the agents are homogeneous. Please see Sec. 4.3 in [2] and
            Sec. 4.1 & 4.2 in [3] for details.
        - False: each actor use independent policy
    - critic_parameter_sharing:
        - True: all actors share a single critic which enables parameters and experiences sharing,
            this is mostly useful where the agents are homogeneous and reward sharing holds. Please
            see Sec. 4.1 in [3] for details.
        - False: each actor use independent critic (though each critic can take other agents actions
            as input, see MADDPG in [1] for details)

    Reference:
    [1] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
    [2] Cooperative Multi-Agent Control Using Deep Reinforcement Learning
    [3] Parameter Sharing Deep Deterministic Policy Gradient for Cooperative Multi-agent Reinforcement Learning

    """
    def __init__(self, env, n_agents, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=10,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True, training_strategy="cocurrent",
                 actor_parameter_sharing=False, critic_parameter_sharing=False):
        super(MAA2C, self).__init__(env, state_dim, action_dim,
                 memory_capacity, max_steps,
                 reward_gamma, reward_scale, done_penalty,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda)

        assert training_strategy in ["cocurrent", "centralized"]

        self.n_agents = n_agents
        self.roll_out_n_steps = roll_out_n_steps
        self.training_strategy = training_strategy
        self.actor_parameter_sharing = actor_parameter_sharing
        self.critic_parameter_sharing = critic_parameter_sharing

        self.actors = [ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act) for _ in range(self.n_agents)]

        if self.training_strategy == "cocurrent":
            self.critics = [CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)] * self.n_agents
        elif self.training_strategy == "centralized":
            critic_state_dim = self.n_agents * self.state_dim
            critic_action_dim = self.n_agents * self.action_dim
            self.critics = [CriticNetwork(critic_state_dim, critic_action_dim, self.critic_hidden_size, 1)] * self.n_agents
        if optimizer_type == "adam":
            self.actor_optimizers = [Adam(self.actors[i].parameters(), lr=self.actor_lr) for i in range(self.n_agents)]
            self.critic_optimizers = [Adam(self.critics[i].parameters(), lr=self.critic_lr) for i in range(self.n_agents)]
        elif optimizer_type == "rmsprop":
            self.actor_optimizers = [RMSprop(self.actors[i].parameters(), lr=self.actor_lr) for i in range(self.n_agents)]
            self.critic_optimizers = [RMSprop(self.critics[i].parameters(), lr=self.critic_lr) for i in range(self.n_agents)]


        # tricky and memory consumed implementation of parameter sharing
        if self.actor_parameter_sharing:
            for agent_id in range(1, self.n_agents):
                self.actors[agent_id] = self.actors[0]
                self.actor_optimizers[agent_id] = self.actor_optimizers[0]
        if self.critic_parameter_sharing:
            for agent_id in range(1, self.n_agents):
                self.critics[agent_id] = self.critics[0]
                self.critic_optimizers[agent_id] = self.critic_optimizers[0]

        if self.use_cuda:
            for a in self.actors:
                a.cuda()
            for c in self.critics:
                c.cuda()

    # agent interact with the environment to collect experience
    # 确保奖励结构平衡，例如在interact函数中调整奖励
    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset()
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action = self.exploration_action(self.env_state)
            next_state, reward, done, _ = self.env.step(action)
            done = done[0]
            actions.append([index_to_one_hot(a, self.action_dim) for a in action])
            rewards.append(reward)
            final_state = next_state
            self.env_state = next_state
            if done:
                self.env_state = self.env.reset()
                break

        if not done:
            rewards[-1] += 0.01  # 每一步增加少量奖励

        if done:
            final_r = [0.0] * self.n_agents
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.episode_done = False
            final_action = self.action(final_state)
            one_hot_action = [index_to_one_hot(a, self.action_dim) for a in final_action]
            final_r = self.value(final_state, one_hot_action)

        rewards = np.array(rewards)
        for agent_id in range(self.n_agents):
            rewards[:,agent_id] = self._discount_reward(rewards[:,agent_id], final_r[agent_id])
        rewards = rewards.tolist()
        self.n_steps += 1
        self.memory.push(states, actions, rewards)

    # def compute_distance_to_nearest_garbage(self, state, agent_id): 
    #     state = state.reshape((11, 11))
    #     garbages_indices = [(i, j) for i in range(len(state)) for j in range(len(state[i])) if state[i][j] == 3]
    #     # print("shape of state: ", state.shape)
    #     agent_index = ((math.sqrt(state.shape[1]-1) - 1) / 2, (math.sqrt(state.shape[1]-1) - 1) / 2)
    #     garbages_indices = np.array(garbages_indices)
    #     agent_index = np.array(agent_index)
    #     if garbages_indices.size == 0:
    #         return np.inf  # 如果没有垃圾，返回无穷大
    #     else:
    #         distances = np.linalg.norm(garbages_indices - agent_index, axis=1)
    #         nearest_distance = np.min(distances)
    #         return nearest_distance
        
    # def get_current_state(self, agent_id):
    #     return self.env._get_local_obs(self.env.agents_pos[agent_id])

    # def train(self):

    #     states = self.env.reset()

    #     for _ in range(self.max_steps):
    #         actions = []

    #         # 逐个智能体选择动作
    #         for agent_id in range(self.n_agents):
    #             state_var = to_tensor_var(states[agent_id], self.use_cuda).view(1, -1)
    #             action_log_probs = self.actors[agent_id](state_var)
    #             action = th.multinomial(th.exp(action_log_probs), 1)
    #             actions.append(action.item())

    #         # 执行动作并获得新状态和奖励
    #         new_states, rewards, done, _ = self.env.step(actions)

    #         # 计算每个智能体与最近垃圾的距离
    #         distances = []
    #         for agent_id in range(self.n_agents):
    #             nearest_distance = self.compute_distance_to_nearest_garbage(new_states[agent_id], agent_id)
    #             distances.append(nearest_distance)

    #         # 逐个智能体更新策略
    #         for agent_id in range(self.n_agents):
    #             self.actor_optimizers[agent_id].zero_grad()

    #             # 使用距离作为损失
    #             nearest_distance_var = th.tensor(distances[agent_id], requires_grad=True, dtype=th.float32).view(-1, 1)
    #             if self.use_cuda:
    #                 nearest_distance_var = nearest_distance_var.cuda()

    #             distance_loss = th.mean(nearest_distance_var)
    #             actor_loss = distance_loss

    #             # 反向传播和优化
    #             actor_loss.backward()
    #             if self.max_grad_norm is not None:
    #                 nn.utils.clip_grad_norm(self.actors[agent_id].parameters(), self.max_grad_norm)
    #             self.actor_optimizers[agent_id].step()

    #         # 更新状态
    #         states = new_states

    #         if not self.env.garbages_pos:
    #             break



    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            return
        # print("training!")
        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)
        whole_states_var = states_var.view(-1, self.n_agents*self.state_dim)
        whole_actions_var = actions_var.view(-1, self.n_agents*self.action_dim)

        for agent_id in range(self.n_agents):
            # update actor network
            self.actor_optimizers[agent_id].zero_grad()
            action_log_probs = self.actors[agent_id](states_var[:,agent_id,:])
            entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
            action_log_probs = th.sum(action_log_probs * actions_var[:,agent_id,:], 1)
            if self.training_strategy == "cocurrent":
                values = self.critics[agent_id](states_var[:,agent_id,:], actions_var[:,agent_id,:])
            elif self.training_strategy == "centralized":
                values = self.critics[agent_id](whole_states_var, whole_actions_var)
            advantages = rewards_var[:,agent_id,:] - values.detach()
            pg_loss = -th.mean(action_log_probs * advantages)
            actor_loss = pg_loss - entropy_loss * self.entropy_reg
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.actors[agent_id].parameters(), self.max_grad_norm)
            self.actor_optimizers[agent_id].step()

            # update critic network
            self.critic_optimizers[agent_id].zero_grad()
            target_values = rewards_var[:,agent_id,:]
            if self.critic_loss == "huber":
                critic_loss = nn.functional.smooth_l1_loss(values, target_values)
            else:
                critic_loss = nn.MSELoss()(values, target_values)
            critic_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(self.critics[agent_id].parameters(), self.max_grad_norm)
            self.critic_optimizers[agent_id].step()

            # 打印损失值
            # print(f"Agent {agent_id}, Actor loss: {actor_loss.item()}, Critic loss: {critic_loss.item()}")

    # predict softmax action based on state
    def _softmax_action(self, state):
        state_var = to_tensor_var([state], self.use_cuda)
        softmax_action = np.zeros((self.n_agents, self.action_dim), dtype=np.float64)
        for agent_id in range(self.n_agents):
            softmax_action_var = th.exp(self.actors[agent_id](state_var[:,agent_id,:]))
            if self.use_cuda:
                softmax_action[agent_id] = softmax_action_var.data.cpu().numpy()[0]
            else:
                softmax_action[agent_id] = softmax_action_var.data.numpy()[0]
        return softmax_action

    # predict action based on state, added random noise for exploration in training
    def exploration_action(self, state):
        softmax_action = self._softmax_action(state)
        actions = [0]*self.n_agents
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                     np.exp(-1. * self.n_steps / self.epsilon_decay)
        for agent_id in range(self.n_agents):
            if np.random.rand() < epsilon:
                actions[agent_id] = np.random.choice(self.action_dim)
            else:
                actions[agent_id] = np.argmax(softmax_action[agent_id])
        return actions

    # predict action based on state for execution
    def action(self, state):
        softmax_actions = self._softmax_action(state)
        actions = np.argmax(softmax_actions, axis=1)
        return actions

    # evaluate value
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action_var = to_tensor_var([action], self.use_cuda)
        whole_state_var = state_var.view(-1, self.n_agents*self.state_dim)
        whole_action_var = action_var.view(-1, self.n_agents*self.action_dim)
        values = [0]*self.n_agents
        for agent_id in range(self.n_agents):
            if self.training_strategy == "cocurrent":
                value_var = self.critics[agent_id](state_var[:,agent_id,:], action_var[:,agent_id,:])
            elif self.training_strategy == "centralized":
                value_var = self.critics[agent_id](whole_state_var, whole_action_var)
            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values
    
    def render_pygame(self, env, screen, window_size):
        # Render the environment and get the image
        img = env.render(mode='rgb_array')
        if img is None:
            print("Environment did not return an image.")
            return

        # Check if the image has three channels (RGB)
        if img.ndim != 3 or img.shape[2] != 3:
            print(f"Unexpected image dimensions: {img.shape}")
            img = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
        else:
            # Clip the image values to be between 0 and 255 and ensure correct type
            img = np.clip(img, 0, 255).astype(np.uint8)
            # Scale the image to the window size
            img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            img_surface = pygame.transform.scale(img_surface, window_size)
            # Blit the image surface to the screen
            screen.blit(img_surface, (0, 0))
            # Update the display
            pygame.display.flip()

        # Handle events to allow window to close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def evaluation(self, env, eval_episodes=10, render=False, window_size=(800, 600)):
        rewards = []
        infos = []

        if render:
            pygame.init()
            screen = pygame.display.set_mode(window_size)
            pygame.display.set_caption("Environment Render")
            clock = pygame.time.Clock()

        for i in range(eval_episodes):
            rewards_i = []
            infos_i = []
            state = env.reset()
            done = np.full(self.n_agents, False)

            for _ in range(self.max_steps):
                action = self.action(state)
                state, reward, done, info = env.step(action)
                if render and i == eval_episodes - 1:
                    self.render_pygame(env, screen, window_size)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return 
                    clock.tick(30)

                done = done[0] if isinstance(done, list) else done
                rewards_i.append(reward)
                infos_i.append(info)

                if not self.env.garbages_pos:
                    break

            rewards.append(np.sum(rewards_i))
            infos.append(infos_i)

        if render:
            pygame.quit()

        return rewards, infos

    
if __name__ == "__main__":
    from NN.env import StreetCleaningEnv  # 确保导入正确的环境

    # 初始化参数
    n_agents = 1
    num_garbage = 100

    # 创建环境实例
    env = StreetCleaningEnv(num_agents=n_agents, num_garbage=num_garbage)
    eval_env_map = env.get_initial_map()
    eval_env = StreetCleaningEnv(num_agents=n_agents, num_garbage=num_garbage, fixed_map=eval_env_map, render=True)
    
    # 提取状态和动作的维度
    state_dim = eval_env.observation_space.shape[1]
    # print("state_dim", state_dim)
    # exit()
    action_dim = eval_env.action_space.n
    
    # 创建MAA2C代理实例
    a2c = MAA2C(
        env=eval_env, 
        n_agents=n_agents, 
        state_dim=state_dim, 
        action_dim=action_dim,
        memory_capacity=10000, 
        max_steps=100, 
        roll_out_n_steps=100, 
        reward_gamma=0.99, 
        reward_scale=1.0, 
        done_penalty=None,
        actor_hidden_size=256,  # 增大隐藏层大小
        critic_hidden_size=256,  # 增大隐藏层大小
        actor_output_act=nn.functional.log_softmax, 
        critic_loss="mse",
        actor_lr=0.001,  # 降低学习率
        critic_lr=0.001,  # 降低学习率
        optimizer_type="adam",  # 使用Adam优化器
        entropy_reg=0.01, 
        max_grad_norm=None,  # 暂时禁用梯度裁剪
        batch_size=64,  # 使用较小的批量大小
        episodes_before_train=10,  # 提前开始训练
        epsilon_start=1.0, 
        epsilon_end=0.1, 
        epsilon_decay=1000,  # 延长探索衰减时间
        use_cuda=True, 
        training_strategy="cocurrent", 
        actor_parameter_sharing=True,  # 尝试参数共享
        critic_parameter_sharing=True  # 尝试参数共享
    )
    
    # 初始化存储结果的列表
    episodes = []
    eval_rewards = []

    # 训练循环
    num_episodes = 10000  # 设定需要训练的回合数
    for episode in range(num_episodes):
        a2c.interact()  # 交互以收集经验
        a2c.train()  # 在批量上进行训练
        
        if (episode+1) % 100 == 0:  # 每100回合打印一次结果
            print(f"Episode {episode+1}/{num_episodes} completed")
    
        # 每个回合结束后，评估模型并保存结果
        if (episode+1) % 1000 == 0:  # 每100回合评估一次
            rewards, infos = a2c.evaluation(eval_env, eval_episodes=10, render=True, window_size=(800, 600))
            rewards_mu = np.mean(rewards)
            print("Episode %d, Average Reward %.2f" % (episode+1, rewards_mu))
            time.sleep(1)
            episodes.append(episode+1)
            eval_rewards.append(rewards_mu)
    
    # 将结果转换为numpy数组
    episodes = np.array(episodes)
    eval_rewards = np.array(eval_rewards)

    # 创建output目录，如果它不存在的话
    if not os.path.exists('./output'):
        os.makedirs('./output')
    # 创建目录
    # 获取当前时间并转换为字符串
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"./output/maa2c/{now}"
    os.makedirs(output_dir, exist_ok=True)

    # 保存文件
    np.savetxt(f"{output_dir}/MAA2C_episodes.txt", episodes)
    np.savetxt(f"{output_dir}/MAA2C_eval_rewards.txt", eval_rewards)
    print("eval_rewards:", eval_rewards)

    # 绘制结果
    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.title("MAA2C")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["MAA2C"])
    plt.savefig(f"{output_dir}/MAA2C.png")

    print("end training!")

