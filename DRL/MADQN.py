
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np

from common.Agent import Agent
from common.Model import ActorNetwork
from common.utils import identity, to_tensor_var
from common.Memory import ReplayMemory

import os 
import datetime
import matplotlib.pyplot as plt
import pygame
import time


class MADQN(Agent):
    """
    An agent learned with DQN using replay memory and temporal difference
    - use a value network to estimate the state-action value
    """
    def __init__(self, n_agents, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=10000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=identity, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):
        super(MADQN, self).__init__(env, state_dim, action_dim,
                 memory_capacity, max_steps,
                 reward_gamma, reward_scale, done_penalty,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda)

        self.n_agents = n_agents
        self.actors = [ActorNetwork(self.state_dim, self.actor_hidden_size,
                                    self.action_dim, self.actor_output_act) for _ in range(n_agents)]
        self.memories = [ReplayMemory(memory_capacity) for _ in range(n_agents)]
        self.episode_done = False
        if self.optimizer_type == "adam":
            self.actor_optimizers = [Adam(actor.parameters(), lr=self.actor_lr) for actor in self.actors]
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizers = [RMSprop(actor.parameters(), lr=self.actor_lr) for actor in self.actors]
        if self.use_cuda:
            for actor in self.actors:
                actor.cuda()

    # agent interact with the environment to collect experience
    def interact(self):
        super(MADQN, self)._take_one_step()

    # train on a sample batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            return

        for actor, actor_optimizer, memory in zip(self.actors, self.actor_optimizers, self.memories):
            batch = memory.sample(self.batch_size)
            states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
            actions_var = to_tensor_var(batch.actions, self.use_cuda, "long").view(-1, 1)
            rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)
            next_states_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.state_dim)
            dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)

            # compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken
            current_q = actor(states_var).gather(1, actions_var)

            # compute V(s_{t+1}) for all next states and all actions,
            # and we then take max_a { V(s_{t+1}) }
            next_state_action_values = actor(next_states_var).detach()
            next_q = th.max(next_state_action_values, 1)[0].view(-1, 1)
            # compute target q by: r + gamma * max_a { V(s_{t+1}) }
            target_q = self.reward_scale * rewards_var + self.reward_gamma * next_q * (1. - dones_var)

            # update value network
            actor_optimizer.zero_grad()
            if self.critic_loss == "huber":
                loss = th.nn.functional.smooth_l1_loss(current_q, target_q)
            else:
                loss = th.nn.MSELoss()(current_q, target_q)
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm(actor.parameters(), self.max_grad_norm)
            actor_optimizer.step()

    # 根据状态选择一个带有随机噪声的动作用于训练中的探索
    def exploration_action(self, states):
        actions = []
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                                np.exp(-1. * self.n_steps / self.epsilon_decay)
        for actor, state in zip(self.actors, states):
            if np.random.rand() < epsilon:
                action = np.random.choice(self.action_dim)
            else:
                state_var = to_tensor_var([state], self.use_cuda)
                state_action_value_var = actor(state_var)
                if self.use_cuda:
                    state_action_value = state_action_value_var.data.cpu().numpy()[0]
                else:
                    state_action_value = state_action_value_var.data.numpy()[0]
                action = np.argmax(state_action_value)
            actions.append(action)
        return actions

    # choose an action based on state for execution
    def action(self, states):
        actions = []
        for actor, state in zip(self.actors, states):
            state_var = to_tensor_var([state], self.use_cuda)
            state_action_value_var = actor(state_var)
            if self.use_cuda:
                state_action_value = state_action_value_var.data.cpu().numpy()[0]
            else:
                state_action_value = state_action_value_var.data.numpy()[0]
            action = np.argmax(state_action_value)
            actions.append(action)
        return actions
    
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
            done = np.zeros(self.n_agents)

            while not np.all(done):
                action = self.action(state)
                state, reward, done, info = env.step(action)
                if render:
                    self.render_pygame(env, screen, window_size)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return 
                    clock.tick(30)

                done = done[0] if isinstance(done, list) else done
                rewards_i.append(reward)
                infos_i.append(info)

            rewards.append(np.sum(rewards_i))
            infos.append(infos_i)

        if render:
            pygame.quit()

        return rewards, infos

if __name__ == "__main__":
    from NN.env import StreetCleaningEnv  # 确保导入正确的环境

    # 初始化参数
    n_agents = 10
    num_garbage = 100

    # 创建环境实例
    env = StreetCleaningEnv(num_agents=n_agents, num_garbage=num_garbage)
    eval_env_map = env.get_initial_map()
    eval_env = StreetCleaningEnv(num_agents=n_agents, num_garbage=num_garbage, fixed_map=eval_env_map, render=True)
    
    # 提取状态和动作的维度
    state_dim = env.observation_space.shape[1]
    # print("state_dim", state_dim)
    # exit()
    action_dim = env.action_space.n

    madqn = MADQN(
        n_agents=n_agents,
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        memory_capacity=10000,
        max_steps=10000,
        reward_gamma=0.99,
        reward_scale=1.,
        done_penalty=None,
        actor_hidden_size=32,
        critic_hidden_size=32,
        actor_output_act=identity,
        critic_loss="mse",
        actor_lr=0.1,
        critic_lr=0.1,
        optimizer_type="rmsprop",
        entropy_reg=0.01,
        max_grad_norm=0.5,
        batch_size=10,
        episodes_before_train=1000,
        epsilon_start=0.9,
        epsilon_end=0.01,
        epsilon_decay=200,
        use_cuda=False
    )

    # 初始化存储结果的列表
    episodes = []
    eval_rewards = []

    # 训练循环
    num_episodes = 100000  # 设定需要训练的回合数
    for episode in range(num_episodes):
        madqn.interact()
        if episode >= madqn.episodes_before_train:
            madqn.train()
        if (episode + 1) % 10000 == 0:
            rewards, _ = madqn.evaluation(eval_env, 10, render=True)
            episodes.append(episode + 1)
            eval_rewards_mu, eval_rewards_std = np.mean(rewards), np.std(rewards)
            eval_rewards.append(eval_rewards_mu)
            print("Episode %d, Average Reward %.2f" % (episode + 1, eval_rewards_mu))
            time.sleep(1)
    
    episodes = np.array(episodes)
    print("eval_rewards: ", eval_rewards)
    eval_rewards = np.array(eval_rewards)

    # 创建output目录，如果它不存在的话
    if not os.path.exists('./output'):
        os.makedirs('./output')
    # 创建目录
    # 获取当前时间并转换为字符串
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"./output/madqn/{now}"
    os.makedirs(output_dir, exist_ok=True)

    # 保存训练过程中的回合数和评估奖励
    np.savetxt(f"{output_dir}/episodes.txt", episodes)
    np.savetxt(f"{output_dir}/eval_rewards.txt", eval_rewards)

    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.title("Street Cleaning with MADQN")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["MADQN"])
    plt.savefig(f"{output_dir}/eval_rewards.png")
    
    print("end training!")
