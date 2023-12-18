from Environments.GridWorldBase import GridWorld
import numpy as np
import torch
import os
import config
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from Experiments.BaseExperiment import BaseExperiment

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
debug = True


class TwoWayGridExperiment(BaseExperiment):
    def __init__(self, agent, env, device, params=None):
        if params is None:
            params = {'render': False}
        super().__init__(agent, env)

        self._render_on = params['render']
        self.num_steps_to_goal_list = []
        self.num_samples = 0
        self.device = device
        self.visited_states = np.array([[0, 0, 0, 0]])

    def start(self):
        self.num_steps = 0
        s = self.environment.start()
        obs = self.observationChannel(s)
        self.last_action = self.agent.start(obs)
        return (obs, self.last_action)

    def step(self):
        (reward, s, term) = self.environment.step(self.last_action)
        self.num_samples += 1
        obs = self.observationChannel(s)
        self.total_reward += reward

        if self._render_on and self.num_episodes >= 0:
            self.environment.render()

        if term:
            self.agent.end(reward)
            roat = (reward, obs, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.step(reward, obs)
            roat = (reward, obs, self.last_action, term)

        self.recordTrajectory(roat[1], roat[2], roat[0], roat[3])
        return roat

    def runEpisode(self, max_steps=0):
        is_terminal = False
        self.start()

        while (not is_terminal) and ((max_steps == 0) or (self.num_steps < max_steps)):
            rl_step_result = self.step()
            is_terminal = rl_step_result[3]

        self.num_episodes += 1
        self.num_steps_to_goal_list.append(self.num_steps)
        if debug:
            print("num steps: ", self.num_steps)
        return is_terminal

    def observationChannel(self, s):
        return np.asarray(s)

    def recordTrajectory(self, s, a, r, t):
        pass


class RunExperiment():
    def __init__(self, use_true_model=False):
        self.device = torch.device("cpu")
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(self.device)
        self.use_true_model = use_true_model
        self.results_dir = "Results/"
        self.plots_dir = "Plots/"

    def run_experiment(self, experiment_object_list, result_file_name, detail=None):
        print("Experiment results will be saved in: \n", result_file_name)
        num_runs = config.num_runs
        num_episode = config.num_episode
        max_step_each_episode = config.max_step_each_episode
        self.num_steps_run_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)
        self.rewards_run_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)
        for i, obj in tqdm(enumerate(experiment_object_list)):
            print("---------------------")
            print("This is the case: ", i)

            for r in range(num_runs):
                print("starting runtime ", r + 1)
                random_obstacle_x = 2 #np.random.randint(0, 8)
                random_obstacle_y = 0 #np.random.choice([0, 2])
                env = GridWorld(params={'size': (3, 7), 'init_state': (1, 0), 'state_mode': 'coord',
                                      'obstacles_pos': [(1, 1),(1, 2), (1, 3), (1, 4), (1, 5),
                                                        (random_obstacle_y, random_obstacle_x)],
                                        'icy_pos': [],
                                      'rewards_pos': [(1, 6)], 'rewards_value': [10],
                                      'terminals_pos': [(1, 6)], 'termination_probs': [1],
                                      'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)],
                                      'neighbour_distance': 0,
                                      'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'obstacle_color': [1, 1, 1], 'icy_color': [1, 0, 0],
                                      'transition_randomness': 0.0,
                                      'window_size': (255, 255),
                                      'aging_reward': 0
                                      })

                corrupt_env = GridWorld(params={'size': (3, 7), 'init_state': (1, 0), 'state_mode': 'coord',
                                        'obstacles_pos': [(1, 1),(1, 2), (1, 3), (1, 4), (1, 5)],
                                        'icy_pos': [],
                                        'rewards_pos': [(1, 6)], 'rewards_value': [10],
                                        'terminals_pos': [(1, 6)], 'termination_probs': [1],
                                        'actions': [(0, 1), (1, 0), (0, -1), (-1, 0)],
                                        'neighbour_distance': 0,
                                        'agent_color': [0, 1, 0], 'ground_color': [0, 0, 0], 'icy_color': [1, 0, 0],
                                        'obstacle_color': [1, 1, 1],
                                        'transition_randomness': 0.0,
                                        'window_size': (255, 255),
                                        'aging_reward': 0
                                        })

                true_fw_model = env.fullTransitionFunction
                if self.use_true_model:
                    corrupted_fw_model  = env.fullTransitionFunction
                else:
                    corrupted_fw_model  = corrupt_env.fullTransitionFunction

                # initializing the agent
                agent = obj.agent_class({'env_name': "two_way",
                                         'action_list': env.getAllActions(),
                                         'gamma': 1, 'epsilon_max': 0.9, 'epsilon_min': 0.1, 'epsilon_decay': 200,
                                         'tau': obj.tau,
                                         'model_corruption': obj.model_corruption,
                                         'max_stepsize': obj.vf_step_size,
                                         'model_stepsize': obj.model_step_size,
                                         'reward_function': None,
                                         'goal': None,
                                         'device': self.device,
                                         'model': obj.model,
                                         'true_bw_model': None,
                                         'true_fw_model': true_fw_model,
                                         'corrupted_fw_model': corrupted_fw_model,
                                         'transition_dynamics': None,
                                         'c': obj.c,
                                         'num_iteration': obj.num_iteration,
                                         'simulation_depth': obj.simulation_depth,
                                         'num_simulation': obj.num_simulation,
                                         'uncertainty_pretrained': config.u_pretrained_u_network,
                                         'vf': obj.vf_network, 'dataset': None})

                # initialize experiment
                experiment = TwoWayGridExperiment(agent, env, self.device)
                for e in range(num_episode):
                    if debug:
                        print("starting episode ", e + 1)
                    experiment.runEpisode(max_step_each_episode)
                    self.num_steps_run_list[i, r, e] = experiment.num_steps
                    self.rewards_run_list[i, r, e] = experiment.total_reward

        with open(self.results_dir + result_file_name + '.p', 'wb') as f:
            result = {'num_steps': self.num_steps_run_list,
                    #   'experiment_objs': experiment_object_list,
                      'rewards': self.rewards_run_list, 
                    #   'detail': detail,
                      }
            pickle.dump(result, f)
        f.close()