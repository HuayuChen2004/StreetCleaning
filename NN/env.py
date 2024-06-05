import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from Render.utils import *

HEIGHT = 54
WIDTH = 78
AGENT_SIZE = 1
VISION_SIZE = 10
MAX_EPISODE_LENGTH = 1000
DEFAULT_FILLING_VALUE = 0
HOUSE_SIZE = 6
LITTER_SIZE = 1

class StreetCleaningEnv(gym.Env):
    def __init__(self, num_agents, num_garbage, fixed_map=None, render=False):
        super(StreetCleaningEnv, self).__init__()
        self.num_agents = num_agents
        self.height = HEIGHT
        self.width = WIDTH
        self.num_garbage = num_garbage
        self.fixed_map = fixed_map
        self.is_render = render
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        observation_dim = (2*VISION_SIZE+AGENT_SIZE) ** 2
        self.observation_space = spaces.Box(low=0, high=3, shape=(num_agents, observation_dim), dtype=np.uint8)
        self.max_episode_length = MAX_EPISODE_LENGTH
        self.reset()
        self.initial_map = self.grid.copy()
        
    def reset(self):
        self.agents_pos = []
        self.garbages_pos = []
        if not isinstance(self.fixed_map, np.ndarray):
            self.grid = np.zeros((self.height, self.width), dtype=np.uint8)  # Swap width and height here
            # self.set_houses()
            self.set_agents()
            self.set_garbage()
        else:
            self.grid = self.fixed_map.copy()
            self.agents_pos = [[x, y] for x, y in np.argwhere(self.grid == 2)]
            self.garbages_pos = [[x, y] for x, y in np.argwhere(self.grid == 3)]
        
        self.done = False
        self.current_step = 0
        return self._get_obs()
    
    def _get_local_obs(self, agent_pos):
        x, y = agent_pos
        
        # Create a larger grid by tiling the original grid
        large_grid = np.tile(self.grid, (3, 3))
        
        # Calculate the position in the large grid
        large_x = x + self.height
        large_y = y + self.width
        
        # Extract the observed grid from the large grid
        observed_grid = large_grid[
            large_x - VISION_SIZE:large_x + VISION_SIZE + AGENT_SIZE,
            large_y - VISION_SIZE:large_y + VISION_SIZE + AGENT_SIZE
        ]
        
        return observed_grid.flatten()



    def _get_obs(self):
        return np.array([self._get_local_obs(agent_pos) for agent_pos in self.agents_pos])
    
    def step(self, action: list):
        rewards = np.zeros(self.num_agents)
        for i, agent_pos in enumerate(self.agents_pos):
            x, y = agent_pos
            if action[i] == 0:  # up
                y = (y - 1) % self.width
            elif action[i] == 2:  # down
                y = (y + 1) % self.width
            elif action[i] == 1:  # left
                x = (x - 1) % self.height
            elif action[i] == 3:  # right
                x = (x + 1) % self.height

            if self.grid[x][y] != 1:
                if self.grid[x][y] == 3:
                    self.garbages_pos.remove([x, y])
                    rewards[i] += 5
                self.grid[agent_pos[0]][agent_pos[1]] = 0
                self.grid[x][y] = 2
                self.agents_pos[i] = [x, y]

        self.done = np.full(self.num_agents, len(self.garbages_pos) == 0)
        self.current_step += 1
        if self.current_step >= self.max_episode_length:
            self.done = np.full(self.num_agents, True)
        return self._get_obs(), rewards, self.done, {}
            
    
    def set_houses(self):
        for x in range(0, self.width, HOUSE_SIZE * 2):  # Change to width here
            for y in range(0, self.height, HOUSE_SIZE * 2):  # Change to height here
                self.grid[y:y+HOUSE_SIZE, x:x+HOUSE_SIZE] = 1  # Swap x and y here
                
    def set_agents(self):
        count = 0
        while count < self.num_agents:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.grid[y][x] == 0:  # Swap x and y here
                self.grid[y][x] = 2  # Swap x and y here
                count += 1
        self.agents_pos = np.array([[y, x] for y, x in np.argwhere(self.grid == 2)])  # Swap x and y here
        
    def set_garbage(self):
        count = 0
        self.garbages_pos = []
        while count < self.num_garbage:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            if self.grid[y][x] == 0:  # Swap x and y here
                self.grid[y][x] = 3  # Swap x and y here
                self.garbages_pos.append([y, x])  # Swap x and y here
                count += 1
        
    def render(self, mode='human', save_path=None):
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255  # Swap width and height here

        img[self.grid == 1] = (128, 128, 128)  # houses
        img[self.grid == 2] = (0, 0, 255)  # agents
        img[self.grid == 3] = (0, 0, 0)  # garbage

        if mode == 'human':
            plt.imshow(img)
            plt.title('Street Cleaning')
            if save_path:
                plt.savefig(save_path)
            plt.show()
            plt.pause(0.1)
            plt.clf()
        elif mode == 'rgb_array':
            return img
        else:
            raise ValueError("mode should be 'human' or 'rgb_array'")
        
    def get_initial_map(self):
        return self.initial_map


if __name__ == "__main__":
    env = StreetCleaningEnv(num_agents=3, num_garbage=20, render=True)
    # map = env.get_initial_map()
    # identical_env = StreetCleaningEnv(num_agents=3, num_garbage=20, fixed_map=map)
    # print("identical:", np.all(env.get_initial_map() == identical_env.get_initial_map()))
    # print("env.agents_pos:", env.agents_pos)
    # print("env.garbages_pos:", env.garbages_pos)
    print("agent places:", np.argwhere(env.grid == 2))
    print("garbage places:", np.argwhere(env.grid == 3))
    # exit()
    env.render()
