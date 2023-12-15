import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import *
from collections import deque
from environments.forever_empty import ForeverEmptyEnv

class Entropygrid:
    def __init__(self, size, max_steps = 100, tile_size = 28, realtime_mode = False, window = 25):
        
        # Set the environment rendering mode
        self._realtime_mode = realtime_mode
        render_mode = "human" if realtime_mode else "rgb_array"
        self.size = size
        #self._env = gym.make(env_name, agent_view_size = 3, tile_size=28, render_mode=render_mode)
        self._env = ForeverEmptyEnv(size=size, render_mode=render_mode, max_steps=max_steps, tile_size = tile_size, has_goal = False)
        self.window = window
        # Decrease the agent's view size to raise the agent's memory challenge
        # On MiniGrid-Memory-S7-v0, the default view size is too large to actually demand a recurrent policy.
        # self._env = RGBImgPartialObsWrapper(self._env, tile_size=28)
        # self._env = ImgObsWrapper(self._env)
        self._observation_space = spaces.Box(
                low = 0,
                high = 1.0,
                shape = (3, tile_size*size, tile_size*size),
                dtype = np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        # This reduces the agent's action space to the only relevant actions (rotate left/right, move forward)
        # to solve the Minigrid-Memory environment.
        return spaces.Discrete(3)

    def reset(self):
        self._rewards = []
        self.pos_history = deque(maxlen=self.window)
        self.time = 1
        obs, _ = self._env.reset(seed=np.random.randint(0, 99))
        obs = obs["image"].astype(np.float32) / 255.
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        
        return obs

    def softreset(self):
        self._rewards = []

    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(action[0])
        self.pos_history.append(self.one_hot(self._env.agent_pos))
        reward = self.entropy(self.pos_history)
        self._rewards.append(reward)
        obs = obs["image"].astype(np.float32) / 255.
        if done or truncated:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        self.time += 1
        
        return obs, reward, done or truncated, info

    def get_history(self):
        return self.pos_history

    def render(self):
        img = self._env.render()
        time.sleep(0.5)
        if len(self.pos_history)>2:
            a = sum(self.pos_history)
            a=a*1/a.sum()
            heat = np.zeros((140,140,3))
            for i in range(140):
                for j in range(140):
                    heat[(i,j,0)] = (int) (100*a[(i//28,j//28)])
            return np.add(2/3*img,1/3*heat)
        return img
        

    def close(self):
        self._env.close()

    def one_hot(self, pair):
        tens = np.zeros((self.size,self.size))
        tens[pair[0]][pair[1]] = 1
        return tens

    def normalize(self, tens_list):
        a = sum(tens_list)
        a=a*1/a.sum()
        return a

    def entropy(self, tens_list):
        # tens_sum = sum(tens_list)
        # a = tens_sum.flatten()
        # a = a/a.sum()
        a=self.normalize(tens_list).flatten()
        ent = 0
        for x in a:
            if x != 0:
                ent += -x*np.log(x)
        return ent