import numpy as np
import gym
from gym import spaces
from Render.utils import *
import cv2

class StreetCleaningEnv(gym.Env):
    def __init__(self, num_agents, num_garbage):
        super(StreetCleaningEnv, self).__init__()
        self.num_agents = num_agents
        self.height = HEIGHT
        self.width = WIDTH
        self.num_garbage = num_garbage
        self.grid = np.zeros((self.width, self.height), dtype=np.uint8)
        self.action_space = spaces.Discrete(5)  # 5 actions: up, down, left, right, clean
        observation_dim = (2*VISION_SIZE+AGENT_SIZE) ** 2
        self.observation_space = spaces.Box(low=0, high=3, shape=(num_agents, observation_dim), dtype=np.uint8)
        self.reset()
        
    def reset(self):
        self.agents_pos = []
        self.garbage_pos = []
        self.grid = np.zeros((self.width, self.height), dtype=np.uint8)
        self.set_houses()
        self.set_agents()
        self.set_garbage()
        self.done = False
        return self._get_obs()
    
    def _get_local_obs(self, agent_pos):
        x, y = agent_pos
        observed_grid = np.full((2*VISION_SIZE+AGENT_SIZE, 2*VISION_SIZE+AGENT_SIZE), fill_value=DEFAULT_FILLING_VALUE)
        
        # Calculate the ranges for the observed grid
        grid_start_x = max(0, VISION_SIZE - x)
        grid_end_x = min(2 * VISION_SIZE + AGENT_SIZE, self.width + VISION_SIZE - x)
        grid_start_y = max(0, VISION_SIZE - y)
        grid_end_y = min(2 * VISION_SIZE + AGENT_SIZE, self.height + VISION_SIZE - y)
        
        # Calculate the ranges for the actual grid
        obs_start_x = max(x - VISION_SIZE, 0)
        obs_end_x = min(x + VISION_SIZE + AGENT_SIZE, self.width)
        obs_start_y = max(y - VISION_SIZE, 0)
        obs_end_y = min(y + VISION_SIZE + AGENT_SIZE, self.height)
        
        # Assign the observed values to the grid
        observed_grid[grid_start_x:grid_end_x, grid_start_y:grid_end_y] = self.grid[obs_start_x:obs_end_x, obs_start_y:obs_end_y]
        # print("local observation shape:", observed_grid.flatten().shape)
        return observed_grid.flatten()

    
    def _get_obs(self):
        return np.array([self._get_local_obs(agent_pos) for agent_pos in self.agents_pos])
    
    
    def step(self, action):
        # multi-agent step
        # action is a list of actions for each agent
        rewards = np.zeros(self.num_agents)
        for i, agent_pos in enumerate(self.agents_pos):
            x, y = agent_pos
            if action[i] == 0 and y > 0 and self.grid[x][y-1] != 1:  # up
                if self.grid[x][y-1] == 3:
                    self.garbage_pos.remove([x, y-1])
                    rewards[i] += 1
                self.grid[x][y] = 0
                self.grid[x][y-1] = 2
                self.agents_pos[i] = [x, y-1]
            elif action[i] == 1 and y < self.height-1 and self.grid[x][y+1] != 1:  # down
                if self.grid[x][y+1] == 3:
                    self.garbage_pos.remove([x, y+1])
                    rewards[i] += 1
                self.grid[x][y] = 0
                self.grid[x][y+1] = 2
                self.agents_pos[i] = [x, y+1]
            elif action[i] == 2 and x > 0 and self.grid[x-1][y] != 1:  # left
                if self.grid[x-1][y] == 3:
                    self.garbage_pos.remove([x-1, y])
                    rewards[i] += 1
                self.grid[x][y] = 0
                self.grid[x-1][y] = 2
                self.agents_pos[i] = [x-1, y]
            elif action[i] == 3 and x < self.width-1 and self.grid[x+1][y] != 1:  # right
                if self.grid[x+1][y] == 3:
                    self.garbage_pos.remove([x+1, y])
                    rewards[i] += 1
                self.grid[x][y] = 0
                self.grid[x+1][y] = 2
                self.agents_pos[i] = [x+1, y]
        self.done = np.full(self.num_agents, len(self.garbage_pos) == 0)
        return self._get_obs(), rewards, self.done, {}
            
    
    def set_houses(self):
        for x in range(0, WIDTH, HOUSE_SIZE * 2):
            for y in range(0, HEIGHT, HOUSE_SIZE * 2):
                self.grid[x:x+HOUSE_SIZE, y:y+HOUSE_SIZE] = 1
                
    def set_agents(self):
        count = 0
        while count < self.num_agents:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.grid[x][y] == 0:
                self.grid[x][y] = 2
                count += 1
        self.agents_pos = np.array([[x, y] for x, y in np.argwhere(self.grid == 2)])
        
    def set_garbage(self):
        count = 0
        self.garbages_pos = []
        while count < self.num_garbage:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.grid[x][y] == 0:
                self.grid[x][y] = 3
                self.garbages_pos.append([x, y])
                count += 1
        self.garbage_pos = np.array(self.garbages_pos)
        
    def render(self, mode='human'):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[self.grid == 1] = (128, 128, 128)  # houses
        img[self.grid == 2] = (0, 0, 255)  # agents
        img[self.grid == 3] = (0, 0, 0)  # garbage
        for i, agent_pos in enumerate(self.agents_pos):
            img[agent_pos[0], agent_pos[1], :] = (0, 0, 255)  # agent positions
        for i, g_pos in enumerate(self.garbage_pos):
            img[g_pos[0], g_pos[1], :] = (0, 0, 0)  # garbage positions
        if mode == 'human':
            cv2.imshow('Street Cleaning', img)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return img
        else:
            raise ValueError("mode should be 'human' or 'rgb_array'")

env = StreetCleaningEnv(num_agents=3, num_garbage=20)
env.reset()
# print("agent positions:", env.agents_pos)
# print("get_observation:", env._get_obs())
# print("observation shape:", env._get_obs().shape)



# 清理需要一定时间
# 1的地方走不了，只能stay

