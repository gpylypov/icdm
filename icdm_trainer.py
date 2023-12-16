import numpy as np
import os
import pickle
import torch
import time
from torch import optim
from buffer import Buffer
from model import ActorCriticModel
from worker import Worker
from utils import create_env
from utils import polynomial_decay
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import wandb

import os
import numpy as np
import torch
import pdb
import trajectory.utils as utils
# import trajectory.datasets as datasets
from trajectory.models.transformers import GPT


class ICDMTrainer:
    def __init__(self, config:dict, run_id:str="run", device:torch.device=torch.device("cpu")) -> None:
        """Initializes all needed training components.

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            run_id {str, optional} -- A tag used to save Tensorboard Summaries and the trained model. Defaults to "run".
            device {torch.device, optional} -- Determines the training device. Defaults to cpu.
        """
        # Set variables
        self.config = config
        self.recurrence = config["recurrence"]
        self.device = device
        self.run_id = config["environment"]["type"]+"-"+run_id
        self.lr_schedule = config["learning_rate_schedule"]
        self.beta_schedule = config["beta_schedule"]
        self.cr_schedule = config["clip_range_schedule"]

        self.args = {'N': 5,
        'action_weight': 5,
        'alpha': 1,
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
        'num_batches': 100,
        'resid_pdrop': 0.1,
        'reward_weight': 1,
        'savepath': 'logs/',
        'inttrajsavepath': 'logs/testing',
        'seed': 42,
        'step': 1,
        'subsampled_sequence_length': 5,
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
        print(self.args['N'])
        block_size = self.args['subsampled_sequence_length'] * transition_dim - 1 + 10
        dataset_size = 300
        print(
            f'Dataset size: {dataset_size} | '
            f'Joined dim: {transition_dim} '
            f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}'
        )

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

        self.model = model_config()
        self.model.to(self.args['device'])

        #######################
        ####### trainer #######
        #######################

        warmup_tokens = dataset_size * block_size ## number of tokens seen per epoch
        final_tokens = 20 * warmup_tokens

        trainer_config = utils.Config(
            utils.Trainer,
            savepath=None, #(self.args['savepath'], 'trainer_config.pkl'),
            # optimization parameters
            batch_size=self.args['batch_size'],
            learning_rate=self.args['learning_rate'],
            betas=(0.9, 0.95),
            grad_norm_clip=1.0,
            weight_decay=0.1, # only applied on matmul weights
            # learning rate decay: linear warmup followed by cosine decay to 10% of original
            lr_decay=self.args['lr_decay'],
            warmup_tokens=warmup_tokens,
            final_tokens=final_tokens,
            ## dataloader
            num_workers=0,
            device=self.args['device'],
            obs_dim = obs_dim,
            act_dim = act_dim,
            discount = self.args['discount'],
            alpha = self.args['alpha'],
            vocab_size=self.args['N'],
            num_batches=self.args['num_batches'],
            num_subsampled_seq = self.args['subsampled_sequence_length'],
            temperature = self.args['temperature']
        )

        self.trainer = trainer_config()

        #######################
        ###### main loop ######
        #######################

        # statepath = os.path.join(self.args['savepath'], f'state_{save_epoch}.pt')
        # print(f'Saving model to {statepath}')

        # ## save state to disk
        # state = model.state_dict()
        # torch.save(state, statepath)


        run = wandb.init(
            # Set the project where this run will be logged
            project="my-awesome-project",
            # Track hyperparameters and run metadata
            config=self.config,
            job_type="icdm-"+config["environment"]["type"],
            mode="offline"
        )


        # Init dummy environment and retrieve action and observation spaces
        print("Step 1: Init dummy environment")
        dummy_env = create_env(self.config["environment"])
        self.observation_space = dummy_env.observation_space
        self.action_space_shape = (dummy_env.action_space.n,)
        dummy_env.close()

        # Init buffer
        print("Step 2: Init buffer")
        self.buffer = Buffer(self.config, self.observation_space, self.action_space_shape, self.device)

        # Init model
        # print("Step 3: Init model and optimizer")
        # self.model = ActorCriticModel(self.config, self.observation_space, self.action_space_shape).to(self.device)
        # self.model.train()
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr_schedule["initial"])

        # Init workers
        print("Step 4: Init environment workers")
        self.workers = [Worker(self.config["environment"]) for w in range(self.config["n_workers"])]

        # Setup observation placeholder   
        self.obs = np.zeros((self.config["n_workers"], 5), dtype=np.float32)

        # Setup initial recurrent cell states (LSTM: tuple(tensor, tensor) or GRU: tensor)
        # hxs, cxs = self.model.init_recurrent_cell_states(self.config["n_workers"], self.device)
        # if self.recurrence["layer_type"] == "gru":
        #     self.recurrent_cell = hxs
        # elif self.recurrence["layer_type"] == "lstm":
        #     self.recurrent_cell = (hxs, cxs)

        # Reset workers (i.e. environments)
        print("Step 5: Reset workers")
        for worker in self.workers:
            worker.child.send(("reset", None))
        # Grab initial observations and store them in their respective placeholder location
        for w, worker in enumerate(self.workers):
            self.obs[w] = worker.child.recv()

    def run_training(self) -> None:
        """Runs the entire training logic from sampling data to optimizing the model."""
        print("Step 6: Starting training")
        ## scale number of epochs to keep number of updates constant
        n_epochs = self.args["n_epochs"]

        num_workers=0
        for w, worker in enumerate(self.workers):
            num_workers+=1
        
        # self.obs
        # breakpoint()
        # context[num_workers by 5]
        print("MAX SEQ LENGTH:", self.args['subsampled_sequence_length'])
        context={"satokens": torch.zeros((num_workers, 1), dtype=torch.long).to(self.args["device"]),
                "rewards": torch.zeros((num_workers, 1), dtype=torch.long).to(self.args["device"])}
        print(context["satokens"].shape)
        print(context["rewards"].shape)
        for update in range(self.config["updates"]):
            # Decay hyperparameters polynomially based on the provided config
            print("Update:", update)
            # context_start_index=(context["satokens"].shape)[1]-6*0 #100
            for t in range(self.config["worker_steps"]):
                # assert self.config["worker_steps"]%100==0
                # if(t%100==0):
                #     if(update==0 and t==0):
                #         context_start_index=1
                #     elif(update==0 and t==100):
                #         context_start_index=context_start_index
                #     elif(t==0):
                #         context_start_index=(context["satokens"].shape)[1]-6*100
                #     else:
                #         context_start_index+=6*100
                #     print(context_start_index, (context["satokens"].shape)[1])

                    
                # Gradients can be omitted for sampling training data
                # print(t)
                context["satokens"]=torch.concat((context["satokens"], torch.tensor(self.obs).type(torch.long).to(self.args["device"])), dim=1)
                # print(context["satokens"].shape)
                # if(update==1):
                #     breakpoint()
                with torch.no_grad():
                    if(update==0 and t<self.args['subsampled_sequence_length']):
                        logits = self.model(context["satokens"][:, 1:])[0][:, -1, :]
                    else:
                        logits = self.model(context["satokens"][:, -(6*self.args['subsampled_sequence_length']-1):])[0][:, -1, :]
                    # elif(update==1):
                    #     logits = self.model(context["satokens"][:, -(6*200-1):])[0][:, -1, :]
                    # elif(update>=2):
                    #     logits = self.model(context["satokens"][:, -(6*200-1):])[0][:, -1, :]

                    # print(logits.shape)
                    # print("logits ", logits)
                    actions = torch.zeros((num_workers, 1), dtype=torch.long).to(device=self.args["device"])
                    # print(actions)
                    # print("obs ", torch.tensor(self.obs).type(torch.long).to(self.args["device"]))
                    for w in range(0, num_workers):
                        # print(w)
                        # print(logits)
                        act_dist=torch.distributions.Categorical(logits=logits[w, :3])

                        actions[w]=int(act_dist.sample())
                    # print(actions)
                    context["satokens"] = torch.concat((context["satokens"], actions), dim=1)
                    # print("actions ", actions)

                # Send actions to the environments
                
                for w, worker in enumerate(self.workers):
                    worker.child.send(("step", actions[w].cpu().numpy()))

                rewards = torch.zeros((num_workers, 1)).to(device=self.args["device"])
                # Retrieve step results from the environments
                for w, worker in enumerate(self.workers):
                    # print(w)
                    obs, rewards[w], _, info = worker.child.recv()
                    # print(rewards[w])
                    if info:
                        # # Store the information of the completed episode (e.g. total reward, episode length)
                        # episode_infos.append(info)
                        # Reset agent (potential interface for providing reset parameters)
                        worker.child.send(("softreset", None))
                        # Get data from reset
                        worker.child.recv()
                        # Reset recurrent cell states
                    # Store latest observations
                    self.obs[w] = obs
                context["rewards"]=torch.concat((context["rewards"], rewards), dim=1)

            if(update==0):
                context["satokens"]=context["satokens"][:, 1:]
                context["rewards"]=context["rewards"][:, 1:]


            wandb.log({"training/value_mean": context["rewards"][:, -600:].sum()/(num_workers*600)})
            print("avg_rewards", context["rewards"][:, -600:].sum()/(num_workers*600))

            # breakpoint()
            # breakpoint()
            for epoch in range(n_epochs):
                print(f'\nEpoch: {epoch} / {n_epochs} | {self.args["dataset"]} | {self.args["exp_name"]}')
                self.trainer.train(self.model, context)
            # breakpoint()
            print(context["satokens"].shape)
            print(context["rewards"].shape)
            
            # Free memory
            # del(self.buffer.samples_flat)
            # if self.device.type == "cuda":
            #     torch.cuda.empty_cache()

        # Save the trained model at the end of the training
        self._save_model()

    def _sample_training_data(self) -> list:
        """Runs all n workers for n steps to sample training data.

        Returns:
            {list} -- list of results of completed episodes.
        """
        episode_infos = []
        # Sample actions from the model and collect experiences for training
        for t in range(self.config["worker_steps"]):
            # Gradients can be omitted for sampling training data
            with torch.no_grad():
                # Save the initial observations and recurrentl cell states
                self.buffer.obs[:, t] = torch.tensor(self.obs)
                if self.recurrence["layer_type"] == "gru":
                    self.buffer.hxs[:, t] = self.recurrent_cell.squeeze(0)
                elif self.recurrence["layer_type"] == "lstm":
                    self.buffer.hxs[:, t] = self.recurrent_cell[0].squeeze(0)
                    self.buffer.cxs[:, t] = self.recurrent_cell[1].squeeze(0)

                # Forward the model to retrieve the policy, the states' value and the recurrent cell states
                policy, value, self.recurrent_cell = self.model(torch.tensor(self.obs), self.recurrent_cell, self.device)
                self.buffer.values[:, t] = value

                # Sample actions from each individual policy branch
                actions = []
                log_probs = []
                for action_branch in policy:
                    action = action_branch.sample()
                    actions.append(action)
                    log_probs.append(action_branch.log_prob(action))
                # Write actions, log_probs and values to buffer
                self.buffer.actions[:, t] = torch.stack(actions, dim=1)
                self.buffer.log_probs[:, t] = torch.stack(log_probs, dim=1)

            # Send actions to the environments
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", self.buffer.actions[w, t].cpu().numpy()))

            # Retrieve step results from the environments
            for w, worker in enumerate(self.workers):
                obs, self.buffer.rewards[w, t], self.buffer.dones[w, t], info = worker.child.recv()
                if info:
                    # Store the information of the completed episode (e.g. total reward, episode length)
                    episode_infos.append(info)
                    # Reset agent (potential interface for providing reset parameters)
                    worker.child.send(("reset", None))
                    # Get data from reset
                    obs = worker.child.recv()
                    # Reset recurrent cell states
                    if self.recurrence["reset_hidden_state"]:
                        hxs, cxs = self.model.init_recurrent_cell_states(1, self.device)
                        if self.recurrence["layer_type"] == "gru":
                            self.recurrent_cell[:, w] = hxs
                        elif self.recurrence["layer_type"] == "lstm":
                            self.recurrent_cell[0][:, w] = hxs
                            self.recurrent_cell[1][:, w] = cxs
                # Store latest observations
                self.obs[w] = obs
                            
        # Calculate advantages
        _, last_value, _ = self.model(torch.tensor(self.obs), self.recurrent_cell, self.device)
        self.buffer.calc_advantages(last_value, self.config["gamma"], self.config["lamda"])

        return episode_infos

    def _train_epochs(self, learning_rate:float, clip_range:float, beta:float) -> list:
        """Trains several PPO epochs over one batch of data while dividing the batch into mini batches.
        
        Arguments:
            learning_rate {float} -- The current learning rate
            clip_range {float} -- The current clip range
            beta {float} -- The current entropy bonus coefficient
            
        Returns:
            {list} -- Training statistics of one training epoch"""
        train_info = []
        for _ in range(self.config["epochs"]):
            # Retrieve the to be trained mini batches via a generator
            mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            for mini_batch in mini_batch_generator:
                train_info.append(self._train_mini_batch(mini_batch, learning_rate, clip_range, beta))
        return train_info

    def _train_mini_batch(self, samples:dict, learning_rate:float, clip_range:float, beta:float) -> list:
        """Uses one mini batch to optimize the model.

        Arguments:
            mini_batch {dict} -- The to be used mini batch data to optimize the model
            learning_rate {float} -- Current learning rate
            clip_range {float} -- Current clip range
            beta {float} -- Current entropy bonus coefficient

        Returns:
            {list} -- list of trainig statistics (e.g. loss)
        """
        # Retrieve sampled recurrent cell states to feed the model
        if self.recurrence["layer_type"] == "gru":
            recurrent_cell = samples["hxs"].unsqueeze(0)
        elif self.recurrence["layer_type"] == "lstm":
            recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))

        # Forward model
        policy, value, _ = self.model(samples["obs"], recurrent_cell, self.device, self.buffer.actual_sequence_length)
        
        # Policy Loss
        # Retrieve and process log_probs from each policy branch
        log_probs, entropies = [], []
        for i, policy_branch in enumerate(policy):
            log_probs.append(policy_branch.log_prob(samples["actions"][:, i]))
            entropies.append(policy_branch.entropy())
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)
        
        # Remove paddings
        value = value[samples["loss_mask"]]
        log_probs = log_probs[samples["loss_mask"]]
        entropies = entropies[samples["loss_mask"]] 

        # Compute policy surrogates to establish the policy loss
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8)
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape)) # Repeat is necessary for multi-discrete action spaces
        ratio = torch.exp(log_probs - samples["log_probs"])
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Value  function loss
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = entropies.mean()

        # Complete loss
        loss = -(policy_loss - self.config["value_loss_coefficient"] * vf_loss + beta * entropy_bonus)

        # Compute gradients
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config["max_grad_norm"])
        self.optimizer.step()

        return [policy_loss.cpu().data.numpy(),
                vf_loss.cpu().data.numpy(),
                loss.cpu().data.numpy(),
                entropy_bonus.cpu().data.numpy()]

    def _write_training_summary(self, update, training_stats, episode_result) -> None:
        """Writes to an event file based on the run-id argument.

        Arguments:
            update {int} -- Current PPO Update
            training_stats {list} -- Statistics of the training algorithm
            episode_result {dict} -- Statistics of completed episodes
        """
        # if episode_result:
        #     for key in episode_result:
        #         if "std" not in key:
        #             self.writer.add_scalar("episode/" + key, episode_result[key], update)
        # self.writer.add_scalar("losses/loss", training_stats[2], update)
        # self.writer.add_scalar("losses/policy_loss", training_stats[0], update)
        # self.writer.add_scalar("losses/value_loss", training_stats[1], update)
        # self.writer.add_scalar("losses/entropy", training_stats[3], update)
        # self.writer.add_scalar("training/sequence_length", self.buffer.true_sequence_length, update)
        # self.writer.add_scalar("training/value_mean", torch.mean(self.buffer.values), update)
        # self.writer.add_scalar("training/advantage_mean", torch.mean(self.buffer.advantages), update)
        wandb.log({
            "losses/loss": training_stats[2],
            "losses/policy_loss": training_stats[0],
            "losses/value_loss": training_stats[1],
            "losses/entropy": training_stats[3],
            "training/sequence_length": self.buffer.true_sequence_length,
            "training/value_mean": torch.mean(self.buffer.values),
            "training/advantage_mean": torch.mean(self.buffer.advantages)
        })

    @staticmethod
    def _process_episode_info(episode_info:list) -> dict:
        """Extracts the mean and std of completed episode statistics like length and total reward.

        Arguments:
            episode_info {list} -- list of dictionaries containing results of completed episodes during the sampling phase

        Returns:
            {dict} -- Processed episode results (computes the mean and std for most available keys)
        """
        result = {}
        if len(episode_info) > 0:
            for key in episode_info[0].keys():
                if key == "success":
                    # This concerns the PocMemoryEnv only
                    episode_result = [info[key] for info in episode_info]
                    result[key + "_percent"] = np.sum(episode_result) / len(episode_result)
                result[key + "_mean"] = np.mean([info[key] for info in episode_info])
                result[key + "_std"] = np.std([info[key] for info in episode_info])
        return result

    def _save_model(self) -> None:
        """Saves the model and the used training config to the models directory. The filename is based on the run id."""
        if not os.path.exists("./models"):
            os.makedirs("./models")
        self.model.cpu()
        pickle.dump((self.model.state_dict(), self.config), open("./models/" + self.run_id + ".nn", "wb"))
        print("Model saved to " + "./models/" + self.run_id + ".nn")

    def close(self) -> None:
        """Terminates the trainer and all related processes."""
        try:
            self.dummy_env.close()
        except:
            pass

        # try:
        #     self.writer.close()
        # except:
        #     pass

        try:
            for worker in self.workers:
                worker.child.send(("close", None))
        except:
            pass

        time.sleep(1.0)
        exit(0)