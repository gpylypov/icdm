import torch
from docopt import docopt
from trainer import PPOTrainer
from icdm_trainer import ICDMTrainer
from yaml_parser import YamlParser

def main():
    # Command line arguments via docopt
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help
    
    Options:
        --config=<path>            Path to the yaml config file [default: ./configs/cartpole.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    run_id = options["--run-id"]
    cpu = options["--cpu"]
    # Parse the yaml config file. The result is a dictionary, which is passed to the trainer.
    config = YamlParser(options["--config"]).get_config()

    if not cpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")
    
    print("Device used: ", device)

    trainers={
        "train-ppo": PPOTrainer, #trains with PPO 
        "icdm-training": ICDMTrainer # additionally creates 
    }
    # Initialize the PPO trainer and commence training
    print(config)
    if("training_method" in config):
        trainer = trainers[config["training_method"]](config, run_id=run_id, device=device)
    else:
        trainer = PPOTrainer(config, run_id=run_id, device=device)

    trainer.run_training()
    trainer.close()

if __name__ == "__main__":
    main()