import numpy as np
import pickle
import torch
from docopt import docopt
from model import ActorCriticModel
from utils import create_env
import wandb
import time
from matplotlib import animation
import matplotlib.pyplot as plt
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
        #Mess with this to change frame size
        plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
        anim.save(path + filename, writer='imagemagick', fps=10)
def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        enjoy.py [options]
        enjoy.py --help
    
    Options:
        --model=<path>              Specifies the path to the trained model [default: ./models/minigrid.nn].
    """
    options = docopt(_USAGE)
    model_path = options["--model"]

    # Inference device
    device = torch.device("cpu")
    torch.set_default_tensor_type("torch.FloatTensor")

    # Load model and config
    state_dict, config = pickle.load(open(model_path, "rb"))

    timestamp = time.strftime("/%Y%m%d-%H%M%S" + "/")
    run = wandb.init(
            # Set the project where this run will be logged
            project="my-awesome-project",
            # Track hyperparameters and run metadata
            config=config,
            mode="offline",
            job_type="vis-"+config["environment"]["type"],
            name=timestamp
        )

    # Instantiate environment
    print(config["environment"])
    if(config["environment"]['type']=="Deviategrid"):
        env = create_env(config["environment"])
    else:
        env = create_env(config["environment"], render=True)

    

    # Initialize model and load its parameters
    model = ActorCriticModel(config, env.observation_space, (env.action_space.n,))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    NUM_GIFS=1
    for i in range(0, NUM_GIFS):
        # Run and render episode
        done = False
        episode_rewards = []

        # Init recurrent cell
        hxs, cxs = model.init_recurrent_cell_states(1, device)
        if config["recurrence"]["layer_type"] == "gru":
            recurrent_cell = hxs
        elif config["recurrence"]["layer_type"] == "lstm":
            recurrent_cell = (hxs, cxs)
        obs = env.reset()
        print(env.render())
        print(i)
        # breakpoint()
        env_frames=[env.render()]
        while not done:
            # Render environment
            # env.render()
            # Forward model
            policy, value, recurrent_cell = model(torch.tensor(np.expand_dims(obs, 0)), recurrent_cell, device, 1)
            # Sample action
            action = []
            for action_branch in policy:
                action.append(action_branch.sample().item())
            # Step environment
            obs, reward, done, info = env.step(action)
            # print(obs.shape)
            env_frames.append(env.render())
            wandb.log({
                "reward": reward
            })
            episode_rewards.append(reward)
        # print(env.render())
        # breakpoint()

        save_frames_as_gif(env_frames, path="./animations/", filename=config['environment']['type']+"-"+str(i)+'.gif')
    # After done, render last state
    # print(env.render())
    # wandb.log({"end_state:", wandb.Image(env.render())})
    # env.render()

    print("Episode length: " + str(info["length"]))
    print("Episode reward: " + str(info["reward"]))

    env.close()

if __name__ == "__main__":
    main()