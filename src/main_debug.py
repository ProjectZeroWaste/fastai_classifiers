from model_factory import get_learner
from train_factory import train_n_runs
import torch 
from fastai.callbacks import *
from fastai.vision import *

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    from configs.test_cfg import enet_x1 as train_cfg 
    torch.cuda.set_device(train_cfg.cuda_id)
    defaults.device = torch.device(f"cuda:{train_cfg.cuda_id}")
    ## Args Parse initialized
    n_runs, epochs, model, experiment_name = train_cfg.nruns,  train_cfg.epochs, train_cfg.model, train_cfg.experimentname
    print(f"Experiment name {experiment_name}")
    print(f"Model training {model}, with {n_runs} training runs...")
    train_n_runs(n_runs, epochs, cfg=train_cfg, experiment_name=experiment_name, model_name=model)