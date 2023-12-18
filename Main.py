'''
train a DQN with MCTS
See if DQN agrees with DQN (saving the best path)
At what step DQN starts to work with MCTS
'''
# use both mcts trajectories and dqn trajectories in the dqn buffer
# use the selection path but with all children in the path
# rollout with dqn policy in mcts
import argparse

import config

from Experiments.ExperimentObject import ExperimentObject
from Experiments.TwoWayGridExperiment import RunExperiment as TwoWayGrid_RunExperiment
from Experiments.TwoWayGridIcyExperiment import RunExperiment as TwoWayGridIcy_RunExperiment
from Experiments.MinAtarExperiment import RunExperiment as MinAtar_RunExperiment

from Agents.SemiOnlineUAMCTS import *
from Agents.SemiOnlineUAMCTSTwoWay import *
from Agents.MCTSAgentTwoWayOnlineModel import *
from Agents.DynaAgent import *
from Agents.MCTSAgentTwoWayOnlineResidualModel import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--scenario', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=True)
    parser.add_argument('--selection', default=False, action="store_true")
    parser.add_argument('--expansion', default=False, action="store_true")
    parser.add_argument('--simulation', default=False, action="store_true")
    parser.add_argument('--backpropagation', default=False, action="store_true")
    parser.add_argument('--num_run', type=int, default=1)
    parser.add_argument('--num_episode', type=int, default=10)
    parser.add_argument('--ni', type=int, default=10)
    parser.add_argument('--ns', type=int, default=10)    
    parser.add_argument('--ds', type=int, default=30)
    parser.add_argument('--c', type=float, default=2**0.5)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--learn_transition', default=False, action="store_true")
    parser.add_argument('--learn_residual', default=False, action="store_true")
    parser.add_argument('--use_true_model', default=False, action="store_true")


    args = parser.parse_args()

    config.rollout_idea = None # None, 1, 5
    config.selection_idea = None  # None, 1
    config.backpropagate_idea = None  # None, 1
    config.expansion_idea = None # None, 2

    if not args.learn_transition:
        if args.selection:
            config.selection_idea = 1
        if args.expansion:
            config.expansion_idea = 2
        if args.simulation:
            config.rollout_idea = 5
        if args.backpropagation:
            config.backpropagate_idea = 1
    
    config.num_runs = args.num_run
    config.num_episode = args.num_episode

    if args.learn_transition or args.learn_residual:
        config.st_training = True

    if args.scenario == "offline":
        config.u_training = False
        config.use_perfect_uncertainty = True
    else:
        config.u_training = True
        config.use_perfect_uncertainty = False

    config.num_runs = args.num_run
    config.num_episode = args.num_episode

    result_file_name = args.file_name

    if args.env == "two_way" or args.env == "two_way_icy":
        if args.learn_transition:
            agent_class_list = [MCTSAgentTwoWayOnlineModel]
        elif args.learn_residual:
            agent_class_list = [MCTSAgentTwoWayOnlineResidualModel]
        else:
            agent_class_list = [SemiOnlineUAMCTSTwoWay]
    else:
        agent_class_list = [SemiOnlineUAMCTS]

    s_vf_list = config.s_vf_list
    s_md_list = config.s_md_list
    model_corruption_list = config.model_corruption_list
    experiment_detail = config.experiment_detail
   
    num_iteration_list = [args.ni] 
    simulation_depth_list = [args.ds]
    num_simulation_list = [args.ns]
    c_list = [args.c]
    tau_list = [args.tau]
    
    model_list = config.model_list

    vf_list = config.trained_vf_list

    if args.env == "two_way":
        experiment = TwoWayGrid_RunExperiment(args.use_true_model)
    elif args.env == "two_way_icy":
        experiment = TwoWayGridIcy_RunExperiment(args.use_true_model)
    else:
        experiment = MinAtar_RunExperiment(args.env, args.use_true_model)

    experiment_object_list = []
    for agent_class in agent_class_list:
        for s_vf in s_vf_list:
            for model in model_list:
                for vf in vf_list:
                    for s_md in s_md_list:
                        for c in c_list:
                            for num_iteration in num_iteration_list:
                                for simulation_depth in simulation_depth_list:
                                    for num_simulation in num_simulation_list:
                                        for model_corruption in model_corruption_list:
                                            for tau in tau_list:
                                                params = {'pre_trained': None,
                                                        'vf_step_size': s_vf,
                                                        'vf': vf,
                                                        'model': model,
                                                        'model_step_size': s_md,
                                                        'c': c,
                                                        'num_iteration': num_iteration,
                                                        'simulation_depth': simulation_depth,
                                                        'num_simulation': num_simulation,
                                                        'model_corruption': model_corruption,
                                                        'tau': tau,}
                                                obj = ExperimentObject(agent_class, params)
                                                experiment_object_list.append(obj)

    experiment.run_experiment(experiment_object_list, result_file_name=result_file_name, detail=experiment_detail)