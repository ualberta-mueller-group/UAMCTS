# Monte Carlo Tree Search in the Presence of Model Uncertainty
This repository provides implementation for the paper [Monte Carlo Tree Search in the Presence of Transition Uncertainty]() (AAAI 2024).

## Libraries requirements
You need to have installed Python 3.8. You also need to install PyTorch, NumPy, Matplotlib, torchvision, tqdm packages:

```
pip3 install torch torchvision torchaudio
pip3 install matplotlib
pip3 install tqdm
```
You also need to install the **modified** version of the [MinAtar Environments](https://github.com/kenjyoung/MinAtar) than includes the corrupted models. To install the MinAtar Environments, use the following command in terminal:
```
pip install MinAtar/.
```

## Experiments
We run the experiments on a **Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz** processor.
To run the experiments, use the following command:

```
python3 Main.py [-h] --env ENV --scenario SCENARIO --file_name FILE_NAME
               [--selection] [--expansion] [--simulation]
               [--backpropagation] [--num_run NUM_RUN]
               [--num_episode NUM_EPISODE] [--ni NI] [--ns NS] [--ds DS]
               [--c C] [--tau TAU] [--learn_transition] [--use_true_model]
```
- ENV can be "space_invaders", "freeway", "breakout, "two_way", or "two_way_icy".
- SCENARIO can be "online" or "offline".
- FILE_NAME is the name of the file of the result of the experiments which will be saved in Results/ directory. This file contians a dictionary with keys 'num_steps' and 'rewards'.
- Use any of the commands --selection, --expansion, --simulation, or --backpropagation to active the the UA corresponding component.
- NUM_RUN and NUM_EPISODE are the number of runs and episodes.
- NI, NS, and DS are the number of iterations, number of simulations, and depth o simulations respectively.
- C is exploration constant and TAU is uncertainty factor. For the online scenario of UAMCTS, TAU is the initial value of the uncertainty factor. 
- Use use_true_model option if you want the agent to have access to the true model of the environment.
- Use learn_transition option if you want to run an experiment with the MCTS agent that learns the transition function online. This option works only for the "two_way" environment, "online" scenario, and it does not work for the UA components.
- If you want to change the maximum number of episodes, uncertainty function or the transition function networks' hyper-parameter, go to config.py.


## Plots
To plot the result of experiments, use the following command:
```
python3 Analyze.py --scenario SCENARIO --file_name FILE_NAME --plot_name
                  PLOT_NAME --metric METRIC
```
- FILE_NAME is the name of the file of the result of the experiments in the Results/ directory. This file contians a dictionary with keys 'num_steps' and 'rewards'.
- SCENARIO can be "online" or "offline".
- PLOT_NAME is the name of the generated plot. The plot will be saved with name PLOT_NAME.png in the UAMCTS/Plot/ directory.
- METRIC can be "num_steps" or "rewards".

## References
Young, K.; and Tian, T. 2019. MinAtar: An Atari-inspired testbed for thorough and reproducible Reinforcement Learn- ing experiments. arXiv preprint arXiv:1903.03176.
