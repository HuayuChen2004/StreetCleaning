import torch as th
from torch import nn
from torch.optim import Adam, RMSprop
import numpy as np
from common.Agent import Agent
from common.Model import ActorNetwork
from common.utils import identity, to_tensor_var, index_to_one_hot
from common.Memory import ReplayMemory
import os
import datetime
import matplotlib.pyplot as plt
import pygame
import time
import math

class MADQN(Agent):
    def __init__(self, n_agents, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=10000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=64, critic_hidden_size=64,
                 actor_output_act=identity, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=1000,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=1000,
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
        self.target_actors = [ActorNetwork(self.state_dim, self.actor_hidden_size,
                                           self.action_dim, self.actor_output_act) for _ in range(n_agents)]
        self.memories = [ReplayMemory(memory_capacity) for _ in range(n_agents)]
        self.episode_done = False
        if self.optimizer_type == "adam":
            self.actor_optimizers = [Adam(actor.parameters(), lr=self.actor_lr) for actor in self.actors]
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizers = [RMSprop(actor.parameters(), lr=self.actor_lr) for actor in self.actors]
        if self.use_cuda:
            for actor, target_actor in zip(self.actors, self.target_actors):
                actor.cuda()
                target_actor.cuda()

        self.update_target_actors()

    def update_target_actors(self):
        for target_actor, actor in zip(self.target_actors, self.actors):
            target_actor.load_state_dict(actor.state_dict())

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

    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            return

        for actor, target_actor, actor_optimizer, memory in zip(self.actors, self.target_actors, self.actor_optimizers, self.memories):
            batch = memory.sample(self.batch_size)
            states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
            actions_var = to_tensor_var(batch.actions, self.use_cuda, "long").view(-1, 1)
            rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)
            next_states_var = to_tensor_var(batch.next_states, self.use_cuda).view(-1, self.state_dim)
            dones_var = to_tensor_var(batch.dones, self.use_cuda).view(-1, 1)

            current_q = actor(states_var).gather(1, actions_var)
            next_state_action_values = target_actor(next_states_var).detach()
            next_q = th.max(next_state_action_values, 1)[0].view(-1, 1)
            target_q = self.reward_scale * rewards_var + self.reward_gamma * next_q * (1. - dones_var)

            actor_optimizer.zero_grad()
            if self.critic_loss == "huber":
                loss = th.nn.functional.smooth_l1_loss(current_q, target_q)
            else:
                loss = th.nn.MSELoss()(current_q, target_q)
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
            actor_optimizer.step()

        self.update_target_actors()

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
        img = env.render(mode='rgb_array')
        if img is None:
            print("Environment did not return an image.")
            return

        if img.ndim != 3 or img.shape[2] != 3:
            print(f"Unexpected image dimensions: {img.shape}")
            img = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
            img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
            img_surface = pygame.transform.scale(img_surface, window_size)
            screen.blit(img_surface, (0, 0))
            pygame.display.flip()

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
                            raise SystemExit
                    clock.tick(30)
                rewards_i.append(reward)
                infos_i.append(info)
            rewards.append(np.sum(rewards_i))
            infos.append(infos_i)

        if render:
            pygame.quit()

        return np.mean(rewards), infos

    def _discount_reward(self, rewards, final_value):
        discounted_rewards = np.zeros_like(rewards)
        cumulative_reward = final_value
        for t in reversed(range(len(rewards))):
            cumulative_reward = rewards[t] + self.reward_gamma * cumulative_reward
            discounted_rewards[t] = cumulative_reward
        return discounted_rewards

    def value(self, states, one_hot_action):
        # Assuming that target_actors is a list of target networks for each agent
        q_values = []
        
        for target_actor, action, state in zip(self.target_actors, one_hot_action, states):
            # Convert state and action to tensor variables
            state_var = to_tensor_var([state], self.use_cuda)
            action_var = to_tensor_var([np.argmax(action)], self.use_cuda, "long").view(-1, 1)

            # Print shapes for debugging
            # print("shape of state_var: ", state_var.shape)
            # print("shape of action_var: ", action_var.shape)

            # Get Q value from the target network
            action_hat_var = target_actor(state_var)

            # Gather the Q value corresponding to the action
            q_value = action_hat_var.gather(1, action_var).item()

            # Append the Q value to the list
            q_values.append(q_value)

        return q_values




if __name__ == "__main__":
    from NN.env import StreetCleaningEnv

    n_agents = 10
    num_garbage = 100

    env = StreetCleaningEnv(num_agents=n_agents, num_garbage=num_garbage)
    eval_env_map = env.get_initial_map()
    eval_env = StreetCleaningEnv(num_agents=n_agents, num_garbage=num_garbage, fixed_map=eval_env_map, render=True)
    
    state_dim = env.observation_space.shape[1]
    action_dim = env.action_space.n

    madqn = MADQN(
        n_agents=n_agents,
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        memory_capacity=10000,
        max_steps=1000,
        reward_gamma=0.9,
        reward_scale=1.0,
        done_penalty=None,
        actor_hidden_size=256,  # Increase hidden size
        critic_hidden_size=256,  # Increase hidden size
        actor_output_act=identity,
        critic_loss="mse",
        actor_lr=0.0001,  # Try different learning rates
        critic_lr=0.0001,  # Try different learning rates
        optimizer_type="adam",  # Try a different optimizer
        entropy_reg=0.01,
        max_grad_norm=None,
        batch_size=64,  # Try different batch sizes
        episodes_before_train=100,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=2000,
        use_cuda=False
    )



    episodes = []
    eval_rewards = []

    num_episodes = 1000

    for episode in range(num_episodes):
        madqn.interact()
        if episode >= madqn.episodes_before_train:
            madqn.train()
        if (episode + 1) % 100 == 0:
            rewards, _ = madqn.evaluation(eval_env, 10, render=False)
            episodes.append(episode + 1)
            eval_rewards_mu, eval_rewards_std = np.mean(rewards), np.std(rewards)
            eval_rewards.append(eval_rewards_mu)
            print("Episode %d, Average Reward %.2f" % (episode + 1, eval_rewards_mu))
            time.sleep(1)
    
    episodes = np.array(episodes)
    eval_rewards = np.array(eval_rewards)

    if not os.path.exists('./output'):
        os.makedirs('./output')
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = f"./output/madqn/{now}"
    os.makedirs(output_dir, exist_ok=True)

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
