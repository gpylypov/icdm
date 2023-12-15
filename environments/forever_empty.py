from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
import numpy as np


class ForeverEmptyEnv(MiniGridEnv):
    """
    ok so its gonna be this except if u reach the goal the goal just resamples and u keep going forever
    
    ## Description

    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Empty-5x5-v0`
    - `MiniGrid-Empty-Random-5x5-v0`
    - `MiniGrid-Empty-6x6-v0`
    - `MiniGrid-Empty-Random-6x6-v0`
    - `MiniGrid-Empty-8x8-v0`
    - `MiniGrid-Empty-16x16-v0`

    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        tile_size = 28,
        obs_img = True,
        has_goal = True,
        max_steps: int | None = None,
        render_mode: str | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        assert max_steps is not None

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            agent_view_size=size,
            render_mode=render_mode,
            highlight=False,
            **kwargs,
        )
        self.tile_size = tile_size
        self.has_goal = has_goal
        self.obs_img = obs_img
        self.goal_spot = (0,0)

        
    def _reward(self):
        #return 1    
        return 1

    def step(
        self, action: ActType
    ):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                # terminated = True
                reward = self._reward()
                self.grid.set(fwd_pos[0], fwd_pos[1], None)
                i =  np.random.randint(1,self.width-1)
                j =  np.random.randint(1,self.height-1)
                self.put_obj(Goal(), i, j)
                self.goal_spot = (i,j)

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count % self.max_steps == 0:
            truncated = True

        if self.render_mode == "human":
            self.render()
        
        if self.obs_img:
            obs = self.gen_obs()
        else:
            obs = (self.agent_start_pos[0],self.agent_start_pos[1],self.goal_spot[0],self.goal_spot[1],self.agent_dir)
        return obs, reward, terminated, truncated, {}


    def gen_obs(self):
        return {"image": self.get_frame(highlight=None, tile_size=self.tile_size), "direction": self.agent_dir, "mission": self.mission}

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        if self.has_goal:
            i =  np.random.randint(1,self.width-1)
            j =  np.random.randint(1,self.height-1)
            self.put_obj(Goal(), i, j)
            self.goal_spot = (i,j)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"