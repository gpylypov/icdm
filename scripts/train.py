import os
import numpy as np
import torch
import pdb

import trajectory.utils as utils
# import trajectory.datasets as datasets
from trajectory.models.transformers import GPT


# class Parser(utils.Parser):
#     dataset: str = 'halfcheetah-medium-expert-v2'
#     config: str = 'config.offline'

# #######################
# ######## setup ########
# #######################

# args = Parser().parse_args('train')

#######################
####### dataset #######
#######################

# env = datasets.load_environment(args.dataset)

# sequence_length = args.subsampled_sequence_length * args.step

# dataset_config = utils.Config(
#     datasets.DiscretizedDataset,
#     savepath=(args.savepath, 'data_config.pkl'),
#     env=args.dataset,
#     N=args.N,
#     penalty=args.termination_penalty,
#     sequence_length=sequence_length,
#     step=args.step,
#     discount=args.discount,
#     discretizer=args.discretizer,
# )

# dataset = dataset_config()
# obs_dim = dataset.observation_dim
# act_dim = dataset.action_dim
# transition_dim = dataset.joined_dim

args = {'N': 5,
 'action_weight': 5,
 'alpha': 0.05,
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
 'n_epochs': 100,
 'n_epochs_ref': 50,
 'n_head': 4,
 'n_layer': 4,
 'n_saves': 500,
 'num_batches': 10,
 'resid_pdrop': 0.1,
 'reward_weight': 1,
 'savepath': 'logs/',
 'inttrajsavepath': 'logs/testing',
 'seed': 42,
 'step': 1,
 'subsampled_sequence_length': 1000,
 'termination_penalty': -100,
 'value_weight': 1}
print(args)

#######################
######## model ########
#######################
print(args['subsampled_sequence_length'])
obs_dim = 5
act_dim = 1
transition_dim=obs_dim+act_dim
print(args['N'])
block_size = args['subsampled_sequence_length'] * transition_dim - 1 + 10
dataset_size = 300
print(
    f'Dataset size: {dataset_size} | '
    f'Joined dim: {transition_dim} '
    f'(observation: {obs_dim}, action: {act_dim}) | Block size: {block_size}'
)

model_config = utils.Config(
    GPT,
    savepath=None, #(args['savepath'], 'model_config.pkl'),
    ## discretization
    vocab_size=args['N'], block_size=block_size,
    ## architecture
    n_layer=args['n_layer'], n_head=args['n_head'], n_embd=args['n_embd']*args['n_head'],
    ## dimensions
    observation_dim=obs_dim, action_dim=act_dim, transition_dim=transition_dim,
    ## loss weighting
    action_weight=args['action_weight'], reward_weight=args['reward_weight'], value_weight=args['value_weight'],
    ## dropout probabilities
    embd_pdrop=args['embd_pdrop'], resid_pdrop=args['resid_pdrop'], attn_pdrop=args['attn_pdrop'],
)

model = model_config()
model.to(args['device'])

#######################
####### trainer #######
#######################

warmup_tokens = dataset_size * block_size ## number of tokens seen per epoch
final_tokens = 20 * warmup_tokens

trainer_config = utils.Config(
    utils.Trainer,
    savepath=None, #(args['savepath'], 'trainer_config.pkl'),
    # optimization parameters
    batch_size=args['batch_size'],
    learning_rate=args['learning_rate'],
    betas=(0.9, 0.95),
    grad_norm_clip=1.0,
    weight_decay=0.1, # only applied on matmul weights
    # learning rate decay: linear warmup followed by cosine decay to 10% of original
    lr_decay=args['lr_decay'],
    warmup_tokens=warmup_tokens,
    final_tokens=final_tokens,
    ## dataloader
    num_workers=0,
    device=args['device'],
    obs_dim = obs_dim,
    act_dim = act_dim,
    discount = args['discount'],
    alpha = args['alpha'],
    vocab_size=args['N'],
    num_batches=args['num_batches'],
    num_subsampled_seq = args['subsampled_sequence_length']
)

trainer = trainer_config()

#######################
###### main loop ######
#######################

## scale number of epochs to keep number of updates constant
n_epochs = int(1e6 / dataset_size * args['n_epochs_ref'])
n_epochs = args["n_epochs"]
save_freq = int(n_epochs // args['n_saves'])

history = {"satokens": torch.tensor(np.random.choice(5, (3, int(6e4)))),
            "rewards": torch.tensor(np.random.uniform(-1,1, (3, int(1e4))))}


# history = {"satokens": torch.tensor([[0, 4, 1, 3, 2, 1, 3, 3, 4, 2, 4, 0, 4, 3, 3, 4, 4, 0],
#                                     [0, 4, 1, 3, 2, 0, 3, 3, 4, 2, 4, 1, 4, 3, 3, 4, 4, 2],
#                                     [0, 4, 1, 3, 2, 2, 3, 3, 4, 2, 4, 0, 4, 3, 3, 4, 4, 2]]), 
#             "rewards": torch.tensor([[0.5, -0.1, 0.4],
#                                     [0.9, 0.6, -0.5],
#                                     [0.2, -0.2, 0.8]])}
#dataset containing state and action tokens for three transitions
# torch.save(history, args['inttrajsavepath']+"/data.pt")
# context = torch.load(args['inttrajsavepath']+"/data.pt", map_location=lambda storage, loc: storage.cuda(0))
# torch.save(history, "data.pt")
# torch.load("data.pt", map_location=lambda storage, loc: storage.cuda(0))
context={"satokens": history["satokens"].to(args["device"]),
"rewards": history["rewards"].to(args["device"])}

# for update in range(NUM_UPDATES):
#     context <- as much of hitory as fits on gpu
#     additional_context <- collect data with transformer policy using context
#     train autoregressively on context[tokens]+additional_context[tokens] using 
#     empirrically calculateQ-weights discounted from context[rewards]+additional_context[rewards]

# model.load_state_dict(torch.load(args["savepath"]+"/state_3000.pt"))
# breakpoint()
print(context["satokens"].shape)
print(context["rewards"].shape)
for epoch in range(n_epochs):
    print(f'\nEpoch: {epoch} / {n_epochs} | {args["dataset"]} | {args["exp_name"]}')

    # xxx=model(torch.tensor([[1, 2, 3, 0, 4, 5, 4, 3, 2]]).to(args["device"]))
    # print(xxx)
    # print(xxx[0].detach().shape)
    # print(xxx[1])
    trainer.train(model, context)

    ## get greatest multiple of `save_freq` less than or equal to `save_epoch
    # save_epoch = (epoch + 1) // save_freq * save_freq
    # if((epoch+1) == save_epoch):
    #     statepath = os.path.join(args['savepath'], f'state_{save_epoch}.pt')
    #     print(f'Saving model to {statepath}')

    #     ## save state to disk
    #     state = model.state_dict()
    #     torch.save(state, statepath)
# statepath = os.path.join(args['savepath'], f'state_{save_epoch}.pt')
# print(f'Saving model to {statepath}')

# ## save state to disk
# state = model.state_dict()
# torch.save(state, statepath)
