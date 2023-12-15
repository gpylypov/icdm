from environments.cartpole_env import CartPole
from environments.minigrid_env import Minigrid
from environments.deviation_env import Deviategrid
from environments.poc_memory_env import PocMemoryEnv
from environments.memory_gym_env import MemoryGymWrapper
from environments.entropy_env import Entropygrid
from environments.vanilla_goals import Vanillagrid
from environments.mountain_env import Mountaingrid

def create_env(config:dict, render:bool=False):
    """Initializes an environment based on the provided environment name.
    
    Arguments:
        config {dict}: The configuration of the environment.

    Returns:
        {env}: Returns the selected environment instance.
    """
    if config["type"] == "PocMemoryEnv":
        return PocMemoryEnv(glob=False, freeze=True)
    if config["type"] == "CartPole":
        return CartPole(mask_velocity=False)
    if config["type"] == "CartPoleMasked":
        return CartPole(mask_velocity=True, realtime_mode = render)
    if config["type"] == "Minigrid":
        return Minigrid(env_name = config["name"], realtime_mode = render)
    if config["type"] == "Deviategrid":
        return Deviategrid(size = config["size"],  realtime_mode = render, max_steps = config["max_steps"], tile_size = config["tile_size"], obs_img = config["obs_img"])
    if config["type"] == "Vanillagrid":
        return Vanillagrid(size = config["size"],  realtime_mode = render, max_steps = config["max_steps"], tile_size = config["tile_size"], obs_img = config["obs_img"])
    if config["type"] == "Mountaingrid":
        return Mountaingrid(size = config["size"],  realtime_mode = render, max_steps = config["max_steps"], tile_size = config["tile_size"], obs_img = config["obs_img"])
    if config["type"] == "Entropygrid":
        return Entropygrid(size = config["size"],  realtime_mode = render, max_steps = config["max_steps"], tile_size = config["tile_size"], obs_img = config["obs_img"], window = config["window"])
    if config["type"] == "MemoryGym":
        return MemoryGymWrapper(env_name = config["name"], reset_params=config["reset_params"], realtime_mode = render)

def polynomial_decay(initial:float, final:float, max_decay_steps:int, power:float, current_step:int) -> float:
    """Decays hyperparameters polynomially. If power is set to 1.0, the decay behaves linearly. 

    Arguments:
        initial {float} -- Initial hyperparameter such as the learning rate
        final {float} -- Final hyperparameter such as the learning rate
        max_decay_steps {int} -- The maximum numbers of steps to decay the hyperparameter
        power {float} -- The strength of the polynomial decay
        current_step {int} -- The current step of the training

    Returns:
        {float} -- Decayed hyperparameter
    """
    # Return the final value if max_decay_steps is reached or the initial and the final value are equal
    if current_step > max_decay_steps or initial == final:
        return final
    # Return the polynomially decayed value given the current step
    else:
        return  ((initial - final) * ((1 - current_step / max_decay_steps) ** power) + final)