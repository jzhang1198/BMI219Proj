#!/usr/bin/env python3

from src import trainer
from ray import tune
import sys

if __name__ == '__main__':
    path2output = sys.argv[1]

    #define hyperparameter search space. if you would like to specify a single model, use tune.choice() with a list of length one.
    config = {'l1': tune.randint(15,35), 
        'l2': tune.randint(5,20),
        'epochs': tune.choice([10]),
        'batch_size': tune.qrandint(100,10000),
        'lr': tune.loguniform(1e-4,1e-2)}

    no_samples = 3 #define job parameters
    cpus = 1
    gpus = 0
    seed = 42
    
    trainer.tune_model(path2output, seed, no_samples, config, cpus, gpus)