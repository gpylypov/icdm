import math
import torch
from torch.utils.data.dataloader import DataLoader
import pdb
import numpy as np
from torch.nn import functional as F

from .timer import Timer

def to(xs, device):
    return [x.to(device) for x in xs]

class Trainer:

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.n_epochs = 0
        self.n_tokens = 0 # counter used for learning rate decay
        self.state_length=config.obs_dim
        self.action_length=config.act_dim
        self.optimizer = None
        self.discount = config.discount
        self.alpha = config.alpha
        self.vocab_size = config.vocab_size
        self.num_subsampled_seq = config.num_subsampled_seq
        self.num_batches = config.num_batches

    def get_optimizer(self, model):
        if self.optimizer is None:
            print(f'[ utils/training ] Making optimizer at epoch {self.n_epochs}')
            self.optimizer = model.configure_optimizers(self.config)
        return self.optimizer

    def train(self, model, context, n_epochs=1, log_freq=100):

        config = self.config
        optimizer = self.get_optimizer(model)
        model.train(True)
        # vocab_size = dataset.N

        # loader = DataLoader(dataset, shuffle=True, pin_memory=True,
        #                     batch_size=config.batch_size,
        #                     num_workers=config.num_workers)
        satokens = context["satokens"]
        rewards = context["rewards"]
        # print(satokens)
        # print(rewards)
        batch_size=satokens.shape[0]
        num_tokens = satokens.shape[1]
        num_states = num_tokens//(self.state_length+self.action_length)
        # print(num_states)
        # print("num_states: ", num_states)
        # print("batch_size: ", batch_size)


        for _ in range(n_epochs):

            losses = []
            timer = Timer()
            #we are going to load in a sequence of state tokens and action tokens. 
            # for a tensor of indices of length n*(state_length+action_length), 
            # we take [i*(state_length+action_length): j*(state_length+action_length)] splice and train on it
            # num_batches=10
            batches = np.random.choice(max(num_states-self.num_subsampled_seq, 1), self.num_batches)
            # batches = [0, 0, 0]
            print(batches)

            for it, i in enumerate(batches):
                # print("iteration ", it)
                # print("i", i)
                j=i+min(num_states, self.num_subsampled_seq)
                seq_to_be_autoregressed = satokens[:, i*(self.state_length+self.action_length): j*(self.state_length+self.action_length)]
                # seq_to_be_autoregressed = to(satokens[:, i*(self.state_length+self.action_length): j*(self.state_length+self.action_length)], self.device)
                # print("Input sequence shape: ", seq_to_be_autoregressed.shape)
                associated_emp_q_estimates= torch.zeros((batch_size, (j-i))).to(device=self.device)

                #Calculate empirical Q values
                running_estimate=torch.zeros((batch_size, )).to(device=self.device)
                # print(running_estimate.shape)
                # print(associated_emp_q_estimates.shape)
                for ctr in range (0, j-i):
                    running_estimate=self.discount*running_estimate
                    associated_emp_q_estimates[:, -1-ctr]=rewards[:, -1-ctr]+running_estimate
                    running_estimate=associated_emp_q_estimates[:, -1-ctr]
                # print("Q estimates shape: ", associated_emp_q_estimates.shape)

                # batch = to(batch, self.device)
                # the passed in seq_to_be_autoregressed is assumed to be [B x T] of indices in the vocab
                # forward the model
                with torch.set_grad_enabled(True):
                    logits, _ = model(seq_to_be_autoregressed)
                    # print(logits.shape)
                    # print(logits.reshape(-1, logits.size(-1)).shape)
                    # breakpoint()
                    # float_version=seq_to_be_autoregressed[:, 1:].reshape(-1).type(torch.float32)
                    loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), seq_to_be_autoregressed[:, 1:].reshape(-1), reduction='none').reshape(batch_size, -1)
                    # print("Loss shape: ", loss.shape)
                    #weighting by Q values and beta
                    # currently action prediction at self.state_length+k(self.state_length+self.action_length)
                    act_pred_weights=-1*self.alpha*torch.exp(associated_emp_q_estimates)
                    weight_mults = torch.cat((torch.ones(batch_size, j-i, self.state_length).to(device=self.device), associated_emp_q_estimates.reshape(batch_size, j-i, 1)), dim=2).reshape(batch_size, -1)[:, 1:]
                    # print(weight_mults)
                    loss = torch.sum(loss*weight_mults)/(batch_size*((j-i)*(self.state_length+self.action_length)-1))
                    # print(loss)
                    # print(zero_padded.reshape(batch_size, -1))
                    losses.append(loss.item())

                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                # decay the learning rate based on our progress
                if config.lr_decay:
                    y = seq_to_be_autoregressed[-2]
                    self.n_tokens += (y != self.vocab_size).sum() # number of tokens processed this step
                    if self.n_tokens < config.warmup_tokens:
                        # linear warmup
                        lr_mult = float(self.n_tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(self.n_tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                # report progress
                if it % log_freq == 0:
                    print(
                        f'[ utils/training ] epoch {self.n_epochs} [ {it:4d} / {self.num_batches:4d} ] ',
                        f'train loss {loss.item():.5f} | lr {lr:.3e} | lr_mult: {lr_mult:.4f} | '
                        f't: {timer():.2f}')

            self.n_epochs += 1
