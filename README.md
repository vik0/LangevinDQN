Langevin DQN - code    
===================

**Authors:** Vikranth Dwaracherla, Benjamin Van Roy

This is a git respository for the code from our paper [Langevin DQN](https://arxiv.org/abs/2002.07282)

We also include some runnable examples in this repository. 

## Getting started 

You can get started with [this colab tutorial](https://github.com/vik0/LangevinDQN/blob/main/LangevinDQN_tutorial.ipynb). This provides an example of how to use our code and run experiments. 

If you press "open in colab" you will have the opportunity to run this python code direct in your browser (using Google Colaboratory, for free) without needing to install anything. This can be a great way to play around with code before you decide to move anything to your own machine.


You  can follow the instructions below to run the experiments on your local machine.

### Installation

The code makes use of [bsuite](https://github.com/deepmind/bsuite) library. The agent follows the bsuite baselines class. You need to install `bsuite` and its dependecies. 

- **(Optional)** It is recommended to use Phyton virutal environment to manage your dependencies
```
python3 -m venv bsuite_env
source bsuite_env/bin/activate
pip install --upgrade pip setuptools
```

- Install bsuite and dependencies for its [baselines](https://github.com/deepmind/bsuite/tree/master/bsuite/baselines) using the following command:
```
pip install bsuite[baselines]
```

- Clone the LangevinDQN repository 
```
git clone https://github.com/vik0/LangevinDQN.git
```

## Running an experiment 

- Langevin DQN:  `python run_point_estimate.py` 

- Ensemble Langevin DQN: `python run_ensemble_langevin_dqn.py` 

Each of the run files mentioned above has flags which set the values of various hyperparameters. We can change them to run experiments with different hyperparameter settings. For example, if we want to run Ensemble Langevin DQN of size 10 on a deep sea environment of size 10 (Bsuite id - deep_sea/0), we can simply run `python run_ensemble_langevin_dqn.py --bsuite_id=deep_sea/0 --num_ensemble=10`. Alternatively, one can simply edit the run file and run it.
