import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import *
from collections import deque
from environments.forever_empty import ForeverEmptyEnv

class Vanillagrid:
    def __init__(self, size, max_steps = 100, tile_size = 28, realtime_mode = False, window = 25,obs_img=True):
        
        # Set the environment rendering mode
        self._realtime_mode = realtime_mode
        render_mode = "human" if realtime_mode else "rgb_array"
        self.size = size
        #self._env = gym.make(env_name, agent_view_size = 3, tile_size=28, render_mode=render_mode)
        self._env = ForeverEmptyEnv(size=size, render_mode=render_mode, max_steps=max_steps, tile_size = tile_size,obs_img=obs_img)
        self.window = window
        self.obs_img = obs_img
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
        self.time = 1
        obs, _ = self._env.reset(seed=np.random.randint(0, 99))
        obs = obs["image"].astype(np.float32) / 255.
        # To conform PyTorch requirements, the channel dimension has to be first.
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        
        if self.obs_img:
            return obs
        else:
            return (self._env.agent_start_pos[0],self._env.agent_start_pos[1],self._env.goal_spot[0],self._env.goal_spot[1],self._env.agent_dir)
    def softreset(self):
        self._rewards = []

    def step(self, action):
        obs, reward, done, truncated, info = self._env.step(action[0])
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

    def render(self):
        img = self._env.render()
        time.sleep(0.5)
        return img

    def close(self):
        self._env.close()