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
    if(config["environment"]['type'] in {"Deviategrid", "Vanillagrid", "Entropygrid", "Mountaingrid"}):
        env = create_env(config["environment"])
    else:
        env = create_env(config["environment"], render=True)

    

    if(config['training_method']=="icdm-training"):
        self.args = {'N': 5,
        'action_weight': 5,
        'alpha': 100,
        'attn_pdrop': 0.1,
        'batch_size': 256,
        'commit': '13d88cc5c9275361fd1e06559619cea44c744a59 main',
        'config': 'config.offline',
        'dataset': 'halfcheetah-medium-expert-v2',
        'device': 'cuda',
        'discount': 0.99,
        'discretizer': 'QuantileDiscretizer',
        'embd_pdrop': 0.1,
        'exp_name': 'gpt/azure',
        'learning_rate': 0.0006,
        'logbase': 'logs/',
        'lr_decay': True,
        'n_embd': 32,
        'n_epochs': 4,
        'n_epochs_ref': 50,
        'n_head': 4,
        'n_layer': 4,
        'n_saves': 500,
        'num_batches': 20,
        'resid_pdrop': 0.1,
        'reward_weight': 1,
        'savepath': 'logs/',
        'inttrajsavepath': 'logs/testing',
        'seed': 42,
        'step': 1,
        'subsampled_sequence_length': 10,
        'termination_penalty': -100,
        'value_weight': 1,
        'temperature': 1}
        print(self.args)

        #######################
        ######## model ########
        #######################
        print(self.args['subsampled_sequence_length'])
        obs_dim = 5
        act_dim = 1
        transition_dim=obs_dim+act_dim

        model_config = utils.Config(
            GPT,
            savepath=None, #(self.args['savepath'], 'model_config.pkl'),
            ## discretization
            vocab_size=self.args['N'], block_size=block_size,
            ## architecture
            n_layer=self.args['n_layer'], n_head=self.args['n_head'], n_embd=self.args['n_embd']*self.args['n_head'],
            ## dimensions
            observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim,
            ## loss weighting
            action_weight=self.args['action_weight'], reward_weight=self.args['reward_weight'], value_weight=self.args['value_weight'],
            ## dropout probabilities
            embd_pdrop=self.args['embd_pdrop'], resid_pdrop=self.args['resid_pdrop'], attn_pdrop=self.args['attn_pdrop'],
        )
        model = model_config()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        context["satokens"]=torch.concat((context["satokens"], torch.tensor(self.obs).type(torch.long).to(self.args["device"])), dim=1)
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

    else:
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

        save_frames_as_gif(env_frames, path="./animations/", filename=config['environment']['type']+"-"+str(i)+'-icdm.gif')
    # After done, render last state
    # print(env.render())
    # wandb.log({"end_state:", wandb.Image(env.render())})
    # env.render()

    print("Episode length: " + str(info["length"]))
    print("Episode reward: " + str(info["reward"]))

    env.close()

if __name__ == "__main__":
    main()