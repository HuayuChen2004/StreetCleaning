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
        self.observation_space = spaces.Box(low=0, high=3, shape=(num_agents, self.height, self.width), dtype=np.uint8)
        
        self.reset()
        
    def reset(self):
        self.set_houses()
        self.set_agents()
        self.set_garbage()
        self.done = False
        return self._get_obs()
    
    def _get_obs(self):
        return np.concatenate([self.agents_pos, self.garbage_pos], axis=0)
    
    def step(self, action):
        rewards = np.zeros(self.num_agents)
        i = 0  # Assuming agent index is 0
        if action == 0:  # up
            self.agents_pos[i][1] = max(self.agents_pos[i][1] - 1, 0)
        elif action == 1:  # down
            self.agents_pos[i][1] = min(self.agents_pos[i][1] + 1, self.grid_size - 1)
        elif action == 2:  # left
            self.agents_pos[i][0] = max(self.agents_pos[i][0] - 1, 0)
        elif action == 3:  # right
            self.agents_pos[i][0] = min(self.agents_pos[i][0] + 1, self.grid_size - 1)
        elif action == 4:  # clean
            for j, g_pos in enumerate(self.garbage_pos):
                if np.array_equal(self.agents_pos[i], g_pos):
                    rewards[i] += 1
                    self.garbage_pos = np.delete(self.garbage_pos, j, axis=0)
                    break
        self.done = len(self.garbage_pos) == 0
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
