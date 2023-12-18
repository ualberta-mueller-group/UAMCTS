from Environments.MinAtarEnvironment import *
import numpy as np
import torch
import os
import config
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from Experiments.BaseExperiment import BaseExperiment
import config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
debug = True


class MinAtarExperiment(BaseExperiment):
    def __init__(self, agent, env, device, env_name, params=None):
        if params is None:
            params = {'render': False}
        super().__init__(agent, env)
        self.env_name = env_name
        self._render_on = params['render']
        self.device = device

    def start(self):
        self.total_reward = 0
        s = self.environment.start()
        obs = self.observationChannel(s)
        self.last_action = self.agent.start(obs)
        return (obs, self.last_action)

    def step(self):
        (reward, s, term) = self.environment.step(self.last_action[0])
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
        self.num_steps = 0

        while (not is_terminal) and ((max_steps == 0) or (self.num_steps < max_steps)):
            rl_step_result = self.step()
            is_terminal = rl_step_result[3]

        if debug:
            print("num steps: ", self.num_steps, "total reward: ", self.total_reward)
        return is_terminal

    def observationChannel(self, s):
        if self.env_name == "freeway":
            return np.append(np.append(np.asarray(s[1]).flatten(), s[0]), s[2])
        elif self.env_name == "space_invaders":
            tmp = np.append(np.append(s[1].flatten(), s[2].flatten()), s[3].flatten())
            return np.append(np.append(np.append(np.append(np.append(np.append(tmp, s[0]), s[4]), s[5]), s[6]), s[7]), s[8])
        elif self.env_name == "breakout":
            return np.append(np.append(np.append(np.append(np.append(np.append(
            np.append(s[0], s[1])
            , s[2]), s[3]), s[4]), s[5])
            , s[6]), s[7].flatten())
        else:
            raise NotImplementedError ("env name in experiment is wrong!")

    def recordTrajectory(self, s, a, r, t):
        pass

class TrueModel_Freeway():
    def __init__(self, true_model):
        self.true_model = true_model

    def transitionFunction(self, state, action):
        cars = state[:-2].reshape((len(state) - 2) // 4, 4)
        pos = state[-2]
        move_timer = state[-1]
        state = int(pos), cars.tolist(), int(move_timer)
        reward, next_state, is_terminal = self.true_model(state, action, is_corrupted=False)
        next_state = np.append(np.append(np.asarray(next_state[1]).flatten(), next_state[0]), next_state[2])
        return next_state, is_terminal, reward

    def corruptTransitionFunction(self, state, action):
        cars = state[:-2].reshape((len(state) - 2) // 4, 4)
        pos = state[-2]
        move_timer = state[-1]
        state = int(pos), cars.tolist(), int(move_timer)
        reward, next_state, is_terminal = self.true_model(state, action, is_corrupted=True)
        next_state = np.append(np.append(np.asarray(next_state[1]).flatten(), next_state[0]), next_state[2])
        return next_state, is_terminal, reward

class TrueModel_SpaceInvaders():
    def __init__(self, true_model):
        self.true_model = true_model

    def transitionFunction(self, state, action):
        pos = state[-6]
        f_bullet_map = state[0:100].reshape(10, 10)
        e_bullet_map = state[100:200].reshape(10, 10)
        alien_map = state[200:300].reshape(10, 10)
        alien_dir = state[-5]
        enemy_move_interval = state[-4]
        alien_move_timer = state[-3]
        alien_shot_timer = state[-2]
        shot_timer = state[-1]
        state = int(pos), f_bullet_map.numpy(), e_bullet_map.numpy(), alien_map.numpy(), \
                int(alien_dir), int(enemy_move_interval), int(alien_move_timer), int(alien_shot_timer), int(shot_timer)
        reward, next_state, is_terminal = self.true_model(state, action, is_corrupted=False)
        tmp = np.append(np.append(next_state[1].flatten(), next_state[2].flatten()), next_state[3].flatten())
        next_state = np.append(np.append(
            np.append(np.append(np.append(np.append(tmp, next_state[0]), next_state[4]), next_state[5]), next_state[6]),
            next_state[7]), next_state[8])
        return next_state, is_terminal, reward

    def corruptTransitionFunction(self, state, action):
        pos = state[-6]
        f_bullet_map = state[0:100].reshape(10, 10)
        e_bullet_map = state[100:200].reshape(10, 10)
        alien_map = state[200:300].reshape(10, 10)
        alien_dir = state[-5]
        enemy_move_interval = state[-4]
        alien_move_timer = state[-3]
        alien_shot_timer = state[-2]
        shot_timer = state[-1]
        state = int(pos), f_bullet_map.numpy(), e_bullet_map.numpy(), alien_map.numpy(), \
                int(alien_dir), int(enemy_move_interval), int(alien_move_timer), int(alien_shot_timer), int(shot_timer)
        reward, next_state, is_terminal = self.true_model(state, action, is_corrupted=True)

        tmp = np.append(np.append(next_state[1].flatten(), next_state[2].flatten()), next_state[3].flatten())
        next_state = np.append(np.append(
            np.append(np.append(np.append(np.append(tmp, next_state[0]), next_state[4]), next_state[5]), next_state[6]),
            next_state[7]), next_state[8])
        return next_state, is_terminal, reward

class TrueModel_Breakout():
    def __init__(self, true_model):
        self.true_model = true_model

    def transitionFunction(self, state, action):
        ball_y = state[0]
        ball_x = state[1]
        ball_dir = state[2]
        pos = state[3]
        strike = state[4]
        last_x = state[5]
        last_y = state[6]
        brick_map = state[7:].reshape(10, 10)

        state = int(ball_y), int(ball_x), int(ball_dir), int(pos), bool(strike), int(last_x), int(last_y), brick_map.numpy()
        reward, next_state, is_terminal = self.true_model(state, action, is_corrupted=False)
        next_state = np.append(np.append(np.append(np.append(np.append(np.append(
            np.append(next_state[0], next_state[1])
            , next_state[2]), next_state[3]), next_state[4]), next_state[5])
            , next_state[6]), next_state[7].flatten())
        return next_state, is_terminal, reward

    def corruptTransitionFunction(self, state, action):
        ball_y = state[0]
        ball_x = state[1]
        ball_dir = state[2]
        pos = state[3]
        strike = state[4]
        last_x = state[5]
        last_y = state[6]
        brick_map = state[7:].reshape(10, 10)

        state = int(ball_y), int(ball_x), int(ball_dir), int(pos), bool(strike), int(last_x), int(last_y), brick_map.numpy()
        reward, next_state, is_terminal = self.true_model(state, action, is_corrupted=True)
        next_state = np.append(np.append(np.append(np.append(np.append(np.append(
            np.append(next_state[0], next_state[1])
            , next_state[2]), next_state[3]), next_state[4]), next_state[5])
            , next_state[6]), next_state[7].flatten())
        return next_state, is_terminal, reward

class RunExperiment():
    def __init__(self, env_name, use_true_model=False):
        self.device = torch.device("cpu")
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(self.device)
        self.results_dir = "Results/"
        self.plots_dir = "Plots/"
        self.env_name = env_name
        self.use_true_model = use_true_model
        
    def run_experiment(self, experiment_object_list, result_file_name, detail=None):
        print("Experiment results will be saved in: \n", result_file_name)
        num_runs = config.num_runs
        num_episode = config.num_episode
        max_step_each_episode = config.max_step_each_episode
        self.num_steps_run_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)
        self.rewards_run_list = np.zeros([len(experiment_object_list), num_runs, num_episode], dtype=np.int)
        save_uncertainty_buffer = config.save_uncertainty_buffer
        env_name = self.env_name
        uncertainty_pretrained = config.u_pretrained_u_network
        
        for r in range(num_runs):
            print("starting runtime ", r + 1)
            for i, obj in tqdm(enumerate(experiment_object_list)):
                print("---------------------")
                print("This is the case: ", i)

                env = MinAtar(name=env_name)
                
                if env_name == "space_invaders":
                    true_model = TrueModel_SpaceInvaders(env.transitionFunction)
                elif env_name == "freeway":
                    true_model = TrueModel_Freeway(env.transitionFunction)
                elif env_name == "breakout":
                    true_model = TrueModel_Breakout(env.transitionFunction)
                else:
                    raise NotImplementedError("env name is wrong!")
                
                true_fw_model = true_model.transitionFunction
                if self.use_true_model:
                    corrupted_fw_model  = true_model.transitionFunction
                else:
                    corrupted_fw_model  = true_model.corruptTransitionFunction

                action_list = np.asarray(env.getAllActions()).reshape(len(env.getAllActions()), 1)
                # initializing the agent
                agent = obj.agent_class({'env_name': env_name, 
                                         'action_list': action_list,
                                         'gamma': 1.0, 'epsilon_max': 0.9, 'epsilon_min': 0.00, 'epsilon_decay': 200,
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
                                         'corrupted_fw_model': corrupted_fw_model, #corruptTransitionFunction
                                         'transition_dynamics': None,
                                         'c': obj.c,
                                         'num_iteration': obj.num_iteration,
                                         'simulation_depth': obj.simulation_depth,
                                         'num_simulation': obj.num_simulation,
                                         'uncertainty_pretrained': uncertainty_pretrained,
                                         'vf': obj.vf_network, 'dataset': None})

                # initialize experiment
                experiment = MinAtarExperiment(agent, env, self.device, env_name=env_name)
                for e in range(num_episode):
                    if debug:
                        print("starting episode ", e + 1)
                    experiment.runEpisode(max_step_each_episode)
                    self.num_steps_run_list[i, r, e] = experiment.num_steps
                    self.rewards_run_list[i, r, e] = experiment.total_reward

                    if e % 100 == 99:
                        if save_uncertainty_buffer:
                            agent.save_uncertainty_buffer("l=" + str(len(agent.uncertainty_buffer)) + "_e=" + str(e)+"_"+"_r=" + str(r)+ "_" + result_file_name)
                        with open(self.results_dir + result_file_name + '.p', 'wb') as f:
                            result = {'num_steps': self.num_steps_run_list,
                                    'rewards': self.rewards_run_list, 
                                    # 'experiment_objs': experiment_object_list,
                                    # 'detail': detail,
                                    }
                            pickle.dump(result, f)
                        f.close()
            with open(self.results_dir + result_file_name + '.p', 'wb') as f:
                result = {'num_steps': self.num_steps_run_list,
                        'rewards': self.rewards_run_list, 
                        'experiment_objs': experiment_object_list,
                        'detail': detail,}
                pickle.dump(result, f)
            f.close()    
            
    def show_multiple_experiment_result_paper(self, results_file_name_list, exp_names, plot_name, fig_test, axs_test, is_offline=False):
        def find_best_c(num_steps, experiment_objs):
            removed_list = []
            num_steps_avg = np.mean(num_steps, axis=1)
            print(num_steps_avg)
            for counter1, i in enumerate(experiment_objs):
                for counter2, j in enumerate(experiment_objs):
                    if i.num_iteration == j.num_iteration and \
                    i.num_simulation == j.num_simulation and \
                    i.simulation_depth == j.simulation_depth and \
                    i.tau == j.tau and \
                    i.vf_network == j.vf_network and \
                    i.c != j.c:
                        if num_steps_avg[counter1] < num_steps_avg[counter2]:
                            removed_list.append(counter1)
                        elif num_steps_avg[counter1] > num_steps_avg[counter2]:
                            removed_list.append(counter2)
                        else:
                            if counter1 < counter2:
                                removed_list.append(counter1)
                            else:
                                removed_list.append(counter2)
            num_steps = np.delete(num_steps, removed_list, 0)
            experiment_objs_new = []
            for i, obj in enumerate(experiment_objs):
                if i not in removed_list:
                    experiment_objs_new.append(obj)
            return num_steps, experiment_objs_new
        def find_best_tau(num_steps, experiment_objs):
            removed_list = []
            num_steps_avg = np.mean(num_steps, axis=1)
            for counter1, i in enumerate(experiment_objs):
                for counter2, j in enumerate(experiment_objs):
                    if i.num_iteration == j.num_iteration and \
                    i.num_simulation == j.num_simulation and \
                    i.simulation_depth == j.simulation_depth and \
                    i.tau != j.tau:
                        if num_steps_avg[counter1] < num_steps_avg[counter2]:
                            removed_list.append(counter1)
                        elif num_steps_avg[counter1] > num_steps_avg[counter2]:
                            removed_list.append(counter2)
            num_steps = np.delete(num_steps, removed_list, 0)
            experiment_objs_new = []
            for i, obj in enumerate(experiment_objs):
                if i not in removed_list:
                    experiment_objs_new.append(obj)
            return num_steps, experiment_objs_new

        def combine_experiment_result(result_file_name):
            res = {'num_steps': None, 'rewards': None, 'experiment_objs': None, 'detail': None}
            all_files = os.listdir(self.results_dir)
            for file_name in all_files:
                if result_file_name in file_name:
                    with open(self.results_dir + file_name, 'rb') as f:
                        result = pickle.load(f)
                    f.close()
                    if res['num_steps'] is None:
                        res['num_steps'] = result['num_steps']
                    else:
                        res['num_steps'] = np.concatenate([res['num_steps'], result['num_steps']], axis=1)
                    
                    if res['rewards'] is None:
                        res['rewards'] = result['rewards']
                    else:
                        res['rewards'] = np.concatenate([res['rewards'], result['rewards']], axis=1)
                    
                    if res['experiment_objs'] is None:
                        res['experiment_objs'] = result['experiment_objs']
            return res

        def make_smooth(runs, s=5):
            smooth_runs = np.zeros([runs.shape[0], runs.shape[1] - s])
            for i in range(runs.shape[0]):
                for j in range(runs.shape[1] - s):
                    smooth_runs[i, j] = np.mean(runs[i, j: j + s])
            return smooth_runs

        def offline(plot_name, fig_test, axs_test):
            if len(results_file_name_list) != len(exp_names):
                print("experiments and names won't match", len(results_file_name_list), len(exp_names))
                return None

            # fig_test, axs_test = plt.subplots(1, 1, constrained_layout=True)
            avg_rewards_list = [] 
            std_rewards_list = []

            reward_list = []
            name_list = []
            for i in range(len(results_file_name_list)):
                result_file_name = results_file_name_list[i]
                exp_name = exp_names[i]
                result = combine_experiment_result(result_file_name)

                rewards, experiment_objs = result['rewards'], result['experiment_objs']
                rewards, experiment_objs = find_best_c(rewards, experiment_objs)
                rewards, experiment_objs = find_best_tau(rewards, experiment_objs)
                
                dqn_std = np.array([np.std(rewards, axis=0)])
                rewards = np.array([np.mean(rewards, axis=0)])
                experiment_objs = [experiment_objs[0]]

                for j in experiment_objs:
                    print(exp_names[i],": ","--C: ", round(j.c, 1), "--Tau: ", j.tau, "--AvgReward:", np.mean(rewards[0], axis=0))            
                    print('-------------------------------------------------------------')
                    print('-------------------------------------------------------------')
                    print('-------------------------------------------------------------')

                names = experiment_obj_to_name(experiment_objs)
                for i, name in enumerate(names):
                    rewards_avg = np.mean(rewards[i], axis=0)
                    rewards_std = np.std(rewards[i], axis=0)
                    if len(rewards_avg) == 1:
                        color = generate_hex_color()
                        if 'C' == exp_name:
                            axs_test.scatter(y=rewards_avg, x=exp_name+name, color="#e0030c")
                            axs_test.axhline(rewards_avg, label=exp_name+name, color="#e0030c", linestyle="--")
                            errorbar = axs_test.errorbar(y=rewards_avg, x=[exp_name+name], yerr=rewards_std,
                                                ls='none', color="#e0030c", capsize=5)

                        elif 'T' == exp_name:
                            axs_test.scatter(y=rewards_avg, x=exp_name+name, color="#06da87")
                            axs_test.axhline(rewards_avg, label=exp_name+name, color="#06da87", linestyle="--")
                            errorbar = axs_test.errorbar(y=rewards_avg, x=[exp_name+name], yerr=rewards_std,
                                                ls='none', color="#06da87", capsize=5)

                        elif 'dqn' in exp_name:
                            axs_test.scatter(y=rewards_avg, x=exp_name+name, color=color)
                            rewards_std = np.mean(dqn_std, axis=1)[0]
                            axs_test.axhline(rewards_avg, label=exp_name+name, color=color, linestyle="-.")
                            errorbar = axs_test.errorbar(y=rewards_avg, x=[exp_name+name], yerr=rewards_std,
                                                linestyle='none', color=color, capsize=5)
                            errorbar[-1][0].set_linestyle("-.")

                        elif '1000' in exp_name:
                            axs_test.scatter(y=rewards_avg, x=exp_name+name, color="#fb9518")
                            # axs_test.axhline(rewards_avg, label=exp_name+name, color="#06da87")
                            errorbar = axs_test.errorbar(y=rewards_avg, x=[exp_name+name], yerr=rewards_std,
                                                ls='none', color="#fb9518", capsize=5)

                        elif '3000' in exp_name:
                            axs_test.scatter(y=rewards_avg, x=exp_name+name, color="#a857b8")
                            # axs_test.axhline(rewards_avg, label=exp_name+name, color="#06da87")
                            errorbar = axs_test.errorbar(y=rewards_avg, x=[exp_name+name], yerr=rewards_std,
                                                ls='none', color="#a857b8", capsize=5)
                            # errorbar[-1][0].set_linestyle("--")                            
                        else:
                            avg_rewards_list.append(np.mean(rewards[i, :, 0]))
                            std_rewards_list.append(np.std(rewards[i, :, 0]))
                            name_list.append(exp_name+name)

            # axs_test.set_xticklabels(name_list, rotation=20, fontsize=8)
            axs_test.scatter(y=avg_rewards_list, x=name_list, color="#256fba")
            axs_test.errorbar(y=avg_rewards_list, x=name_list, yerr=std_rewards_list,
            ls='none', color="#256fba", solid_capstyle='projecting', capsize=5)
            # axs_test.xaxis.set_tick_params(labeltop=True)
            # axs_test.boxplot(reward_list)

            # axs_test.legend()
            # fig_test.savefig(self.plots_dir + plot_name +".png", format="png")
            # fig_test.savefig(self.plots_dir + plot_name +".svg", format="svg")

      
        def online(plot_name, fig_test, axs_test):
            print(results_file_name_list)
            if len(results_file_name_list) != len(exp_names):
                print("experiments and names won't match", len(results_file_name_list), len(exp_names))
                return None
           
            name_list = []

            for i in range(len(results_file_name_list)):
                result_file_name = results_file_name_list[i]
                exp_name = exp_names[i]
                result = combine_experiment_result(result_file_name)
                rewards, experiment_objs = result['rewards'], result['experiment_objs']
                rewards, experiment_objs = find_best_c(rewards, experiment_objs)
                rewards, experiment_objs = find_best_tau(rewards, experiment_objs)
                for j in experiment_objs:
                    if rewards.shape[2] == 1:
                        print(exp_names[i],": ","--C: ", round(j.c, 1), "--Tau: ", j.tau, "--AvgReward:", np.mean(rewards[0], axis=0))            
                    else:
                        print(exp_names[i],": ","--C: ", round(j.c, 1), "--Tau: ", j.tau)


                if(rewards.shape[2] > 1):
                    rewards = np.array([make_smooth(rewards[0], s=10)])

                names = experiment_obj_to_name(experiment_objs)
                for i, name in enumerate(names):
                    rewards_avg = np.mean(rewards[i], axis=0)
                    rewards_std = np.std(rewards[i], axis=0)
                    x = range(len(rewards_avg))
                    if len(rewards_avg) == 1:
                        color = generate_hex_color()
                        if "T" in exp_name:
                            
                            name_list.append(exp_name+name)
                            axs_test.axhline(rewards_avg, label=exp_name+name, color=color, linestyle="--")
                        else:
                            
                            name_list.append(exp_name+name)
                            axs_test.axhline(rewards_avg, label=exp_name+name, color=color, linestyle="-")
                    else:
                        axs_test.plot(x, rewards_avg, label=exp_name+name, color="orchid", )
                        axs_test.fill_between(x,
                                        rewards_avg - rewards_std,
                                        rewards_avg + rewards_std, color="orchid",
                                        alpha=.3, edgecolor='none')
            # axs_test.legend()
            # fig_test.savefig(self.plots_dir + plot_name +".png", format="png")
            # fig_test.savefig(self.plots_dir + plot_name +".svg", format="svg")


        def pretrained_online():
            space = "MinAtarResult/Paper/SpaceInvaders_CorruptedStates=[2, 3, 4, 5, 6]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1.p"
            freeway = "MinAtarResult/Paper/Freeway_CorruptedStates=[1, 2, 3, 5, 6, 7]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1.p"
            breakout = "MinAtarResult/Paper/Breakout_CorruptedStates=[2, 4]_MCTS_R=N_E=N_S=N_B=N_Buffer_run1.p"
            if len(results_file_name_list) != len(exp_names):
                print("experiments and names won't match", len(results_file_name_list), len(exp_names))
                return None
            with open(breakout, 'rb') as f:
                mcts_result = pickle.load(f)
                mcts_reward = mcts_result['rewards']
                mcts_reward = np.array([make_smooth(mcts_reward[0], s=50)])
                # print(mcts_reward.shape)
                done_rewrads = None
                for i in range(mcts_reward.shape[1]):
                    avg = mcts_reward[0, i].mean()
                    if avg > 0:
                        if done_rewrads is None:
                            done_rewrads = np.array([mcts_reward[0, i]])
                        else:
                            tmp = np.array([mcts_reward[0, i]])
                            done_rewrads = np.concatenate([done_rewrads, tmp], axis=0)
                mcts_reward = np.array([done_rewrads[:, :]])
                # print(mcts_reward.shape)
                mcts_avg_reward = np.mean(mcts_reward, axis=1)[0]
                mcts_std_reward = np.std(mcts_reward, axis=0)[0]
                
                mcts_avg_reward = np.mean(mcts_reward, axis=2)[0].mean()

                # print(mcts_avg_reward.shape)
                mcts_avg_reward = np.repeat(mcts_avg_reward, 90)
                mcts_std_reward = np.repeat(0, 90)

            # print(mcts_reward[0])
            training_start_ep = mcts_avg_reward.shape[0]

            # fig_test, axs_test = plt.subplots(1, 1, constrained_layout=True)

            name_list = []
            for i in range(len(results_file_name_list)):
                result_file_name = results_file_name_list[i]
                exp_name = exp_names[i]
                result = combine_experiment_result(result_file_name)
                # print("reward shape:", result['rewards'].shape)
                rewards, experiment_objs = result['rewards'], result['experiment_objs']
                rewards, experiment_objs = find_best_c(rewards, experiment_objs)
                rewards, experiment_objs = find_best_tau(rewards, experiment_objs)
                for j in experiment_objs:
                    if rewards.shape[2] == 1:
                        print(exp_names[i], ": ", "--C: ", round(j.c, 1), "--Tau: ", j.tau, "--AvgReward:",
                              np.mean(rewards[0], axis=0))
                    else:
                        print(exp_names[i], ": ", "--C: ", round(j.c, 1), "--Tau: ", j.tau)
                if True:
                    #only pick the first parameters in case of equal performance
                    rewards = np.array([rewards[0]])
                    experiment_objs = [experiment_objs[0]]

                # done_rewrads = None
                # for i in range(rewards.shape[1]):
                #     avg = rewards[0, i].mean()
                #     if avg > 0:
                #         if done_rewrads is None:
                #             done_rewrads = np.array([rewards[0, i]])
                #         else:
                #             tmp = np.array([rewards[0, i]])
                #             done_rewrads = np.concatenate([done_rewrads, tmp], axis=0)
                # rewards = np.array([done_rewrads[:, :120]])
                # rewards = np.array([make_smooth(rewards[0], s=2)])
                # np.concatenate((rewards, mcts_reward))

                names = experiment_obj_to_name(experiment_objs)
                for i, name in enumerate(names):
                    rewards_avg = np.mean(rewards[i], axis=0)
                    rewards_std = np.std(rewards[i], axis=0)

                    if len(rewards_avg) == 1:

                        color = generate_hex_color()
                        # print(rewards_avg, rewards_std, exp_name + name, "\n")
                        if exp_name == "C":
                            axs_test.axhline(rewards_avg, label=exp_name + name, color=color, linestyle=":")
                        elif exp_name == "T":
                            axs_test.axhline(rewards_avg, label=exp_name + name, color=color, linestyle="--")

                    else:
                        print(rewards_avg.shape, mcts_avg_reward.shape, "***")
                        
                        rewards_avg = np.concatenate((mcts_avg_reward, rewards_avg))
                        rewards_std = np.concatenate((mcts_std_reward, rewards_std))
                        x = range(len(rewards_avg))

                        print(rewards_avg.mean(), rewards_std.mean(), exp_name + name, "\n")

                        axs_test.plot(x, rewards_avg, label=exp_name + name)
                        axs_test.fill_between(x,
                                              rewards_avg -  0.5 * rewards_std,
                                              rewards_avg +  0.5 * rewards_std,
                                              alpha=.4, edgecolor='none')

            ymin = axs_test.get_ylim()[0]
            axs_test.vlines(training_start_ep-1, color="black", linestyle=":", ymax=mcts_avg_reward[-1], ymin=axs_test.get_ylim()[0])
            

            axs_test.legend()
            axs_test.set_ylim(ymin=ymin)

            fig_test.savefig("preonline_test" + ".png", format="png")

            # fig_test.savefig("SpaceInvaders_TrainedUncertainty_Performance_Correct"+".svg", format="svg")
            # fig_test.savefig("Freeway_TrainedUncertainty_Performance_Correct"+".svg", format="svg")
            fig_test.savefig("Breakout_TrainedUncertainty_Performance_Correct"+".svg", format="svg")
        if is_offline:
            offline(plot_name, fig_test, axs_test)
        else:
            online(plot_name, fig_test, axs_test)
        # pretrained_online()

    def unpaired_t_test(self, result_file_name1, result_file_name2, names=['alg1', 'alg2']):
        def find_best_c(num_steps, experiment_objs):
            removed_list = []
            num_steps_avg = np.mean(num_steps, axis=1)
            # print()
            for counter1, i in enumerate(experiment_objs):
                for counter2, j in enumerate(experiment_objs):
                    if i.num_iteration == j.num_iteration and \
                    i.num_simulation == j.num_simulation and \
                    i.simulation_depth == j.simulation_depth and \
                    i.tau == j.tau and \
                    i.c != j.c:
                        if num_steps_avg[counter1] < num_steps_avg[counter2]:
                            removed_list.append(counter1)
                        elif num_steps_avg[counter1] > num_steps_avg[counter2]:
                            removed_list.append(counter2)
            num_steps = np.delete(num_steps, removed_list, 0)
            experiment_objs_new = []
            for i, obj in enumerate(experiment_objs):
                if i not in removed_list:
                    experiment_objs_new.append(obj)
            return num_steps, experiment_objs_new
        def find_best_tau(num_steps, experiment_objs):
            removed_list = []
            num_steps_avg = np.mean(num_steps, axis=1)
            for counter1, i in enumerate(experiment_objs):
                for counter2, j in enumerate(experiment_objs):
                    if i.num_iteration == j.num_iteration and \
                    i.num_simulation == j.num_simulation and \
                    i.simulation_depth == j.simulation_depth and \
                    i.tau != j.tau:
                        if num_steps_avg[counter1] < num_steps_avg[counter2]:
                            removed_list.append(counter1)
                        else:
                            removed_list.append(counter2)
            num_steps = np.delete(num_steps, removed_list, 0)
            experiment_objs_new = []
            for i, obj in enumerate(experiment_objs):
                if i not in removed_list:
                    experiment_objs_new.append(obj)

            return num_steps, experiment_objs_new
        def independent_ttest(data1, data2, alpha):
            import scipy
            print(data1.shape, data2.shape)
            # calculate means
            mean1, mean2 = np.mean(data1), np.mean(data2)
            # calculate standard errors
            se1, se2 = scipy.stats.sem(data1), scipy.stats.sem(data2)
            # standard error on the difference between the samples
            sed = np.sqrt(se1**2.0 + se2**2.0)
            # calculate the t statistic
            t_stat = (mean1 - mean2) / sed
            # degrees of freedom
            df = len(data1) + len(data2) - 2
            # calculate the critical value
            cv = scipy.stats.t.ppf(1.0 - alpha, df)
            # calculate the p-value
            p = (1.0 - scipy.stats.t.cdf(abs(t_stat), df)) * 2.0
            # return everything
            return t_stat, df, cv, p

        result1 = self.combine_experiment_result(result_file_name1)
        result2 = self.combine_experiment_result(result_file_name2)
        rewards1, experiment_objs1 = find_best_c(result1['rewards'], result1['experiment_objs'])
        rewards1, experiment_objs1 = find_best_tau(rewards1, experiment_objs1)
        rewards2, experiment_objs2 = find_best_c(result2['rewards'], result2['experiment_objs'])
        rewards2, experiment_objs2 = find_best_tau(rewards2, experiment_objs2)

        alpha = 0.05
        t_stat, df, cv, p = independent_ttest(rewards1[0, :, 0], rewards2[0, :, 0], alpha = alpha)
        if p < 0.001:
            if t_stat > 0:
                print(names[0] + " results are HIGHLY stastiscally significant better than " + names[1])
            if t_stat < 0:
                print(names[1] + " results are HIGHLY stastiscally significant better than " + names[0])
        elif p < 0.05:
            if t_stat > 0:
                print(names[0] + " results are stastiscally significant better than " + names[1])
            if t_stat < 0:
                print(names[1] + " results are stastiscally significant better than " + names[0])
        else:
            print("The difference between " + names[0] + " and " + names[1] + " is not stastically significant")

    def multiple_experiments_t_test(self, result_file_name_list, names):
        for i in range(len(result_file_name_list)):
            for j in range(len(result_file_name_list)):
                if i == j:
                    continue
                self.unpaired_t_test(result_file_name_list[i], result_file_name_list[j], [names[i], names[j]])

def experiment_obj_to_name(experiment_objs):
    def all_elements_equal(List):
        result = all(element == List[0] for element in List)
        if (result):
            return True
        else:
            return False

    names = [""] * len(experiment_objs)
    # print(names)
    agent_class = [i.agent_class for i in experiment_objs]
    if not all_elements_equal(agent_class):
        for i in range(len(experiment_objs)):
            names[i] += "agent:" + agent_class[i].name

    pre_trained = [i.pre_trained for i in experiment_objs]
    if not all_elements_equal(pre_trained):
        for i in range(len(experiment_objs)):
            names[i] += "pre_trained:" + str(pre_trained[i])

    model = [i.model for i in experiment_objs]
    if not all_elements_equal(model):
        for i in range(len(experiment_objs)):
            names[i] += "model:" + str(model[i])

    model_step_size = [i.model_step_size for i in experiment_objs]
    if not all_elements_equal(model_step_size):
        for i in range(len(experiment_objs)):
            names[i] += "m_stepsize:" + str(model_step_size[i])

    # dqn params
    vf_network = [i.vf_network for i in experiment_objs]
    if not all_elements_equal(vf_network):
        for i in range(len(experiment_objs)):
            names[i] += "vf:" + str(vf_network[i])

    vf_step_size = [i.vf_step_size for i in experiment_objs]
    if not all_elements_equal(vf_step_size):
        for i in range(len(experiment_objs)):
            names[i] += "vf_stepsize:" + str(vf_step_size[i])

    # mcts params
    c = [i.c for i in experiment_objs]
    if not all_elements_equal(c):
        for i in range(len(experiment_objs)):
            names[i] += "c:" + str(c[i])

    num_iteration = [i.num_iteration for i in experiment_objs]
    if not all_elements_equal(num_iteration):
        for i in range(len(experiment_objs)):
            names[i] += "N_I:" + str(num_iteration[i])

    simulation_depth = [i.simulation_depth for i in experiment_objs]
    if not all_elements_equal(simulation_depth):
        for i in range(len(experiment_objs)):
            names[i] += "S_D:" + str(simulation_depth[i])

    num_simulation = [i.num_simulation for i in experiment_objs]
    if not all_elements_equal(num_simulation):
        for i in range(len(experiment_objs)):
            names[i] += "N_S:" + str(num_simulation[i])

    model_corruption = [i.model_corruption for i in experiment_objs]
    if not all_elements_equal(model_corruption):
        for i in range(len(experiment_objs)):
            names[i] += "M_C:" + str(model_corruption[i])

    tau = [i.tau for i in experiment_objs]
    if not all_elements_equal(tau):
        for i in range(len(experiment_objs)):
            names[i] += "Tau:" + str(tau[i])
    # print(names)
    return names

generate_random_color = False
color_counter = 0
color_list = ['#FF2929', '#19A01D', '#F4D03F', '#FF7F50', '#8E44AD', '#34495E', '#95A5A6', '#5DADE2', '#A2FF00', '#003AFF', '#FF008F']
def generate_hex_color():
    global color_counter
    if generate_random_color:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        c = (r, g, b)
        hex_c = '#%02x%02x%02x' % c
    else:
        hex_c = color_list[color_counter]
        color_counter = (color_counter + 1) % len(color_list)
    return hex_c