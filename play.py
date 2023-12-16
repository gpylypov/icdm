import torch
from docopt import docopt
from trainer import PPOTrainer
from icdm_trainer import ICDMTrainer
from yaml_parser import YamlParser
from environments.entropy_env import Entropygrid

if __name__ == "__main__":
    env = Entropygrid(size = 5,  realtime_mode = False, max_steps = 100, 
        tile_size = 28, obs_img = False, window = 25)
    env.reset()
    breakpoint()