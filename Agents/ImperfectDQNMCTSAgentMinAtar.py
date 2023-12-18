import pickle
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import utils
from Agents.MCTSAgent import MCTSAgent as MCTSAgent
from Agents.DynaAgent import DynaAgent
from DataStructures.Node import Node as Node
from Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN
from Networks.ModelNN.StateTransitionModel import UncertaintyNN
import config

class ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction(DynaAgent, MCTSAgent):
    name = "ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction"
    rollout_idea = config.rollout_idea # None, 1, 5
    selection_idea = config.selection_idea  # None, 1
    backpropagate_idea = config.backpropagate_idea  # None, 1
    expansion_idea = config.expansion_idea # None, 2
    assert rollout_idea in [None, 1, 2, 3, 4, 5] and \
           selection_idea in [None, 1, 2] and \
           backpropagate_idea in [None, 1]  \
           and expansion_idea in [None, 1, 2]

    def __init__(self, params={}):
        self.env = params['env_name'] #space_invaders, freeway, breakout
        self.episode_counter = 0
        self.model_corruption = params['model_corruption']
        self.model_loss = []
        self.time_step = 0

        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]
        self.actions_shape = self.action_list.shape[1:]

        self.gamma = params['gamma']
        self.epsilon = params['epsilon_min']

        self.transition_buffer = []
        self.transition_buffer_size = 2 ** 12

        self.uncertainty_buffer = []
        self.uncertainty_buffer_size = 2 ** 20
        if config.pre_gathered_buffer is not None:
            self.load_buffer(config.pre_gathered_buffer)

        self.minimum_uncertainty_buffer_training = config.minimum_uncertainty_buffer_training
        self._uf = {'network':None,
                    'batch_size': config.U_batch_size,
                    'step_size':config.U_step_size,
                    'layers_type':config.U_layers_type,
                    'layers_features':config.U_layers_features,
                    'training':config.U_training}
        self.u_epoch_training = config.U_epoch_training
        self.pre_trained_Unetwork = params['uncertainty_pretrained']
        self.use_perfect_uncertainty = config.use_perfect_uncertainty
        self._sr = dict(network=None,
                        layers_type=[],
                        layers_features=[],
                        batch_size=None,
                        step_size=None,
                        batch_counter=None,
                        training=False)
        if params['vf'] is not None:
            with open(params['vf'], "rb") as file:
                self.value_function = pickle.load(file)
                self.value_function['q']['network'].to("cpu")
        # if config.load_vf is not None:
        #     with open(config.load_vf, "rb") as file:
        #         self.value_function = pickle.load(file)
        #         self.value_function['q']['network'].to("cpu")

        self.device = params['device']
        self.true_model = params['true_fw_model']
        self.corrupt_model = params['corrupted_fw_model']

        self.num_steps = 0
        self.num_terminal_steps = 0

        # MCTS parameters
        self.C = params['c']
        self.num_iterations = params['num_iteration']
        self.num_rollouts = params['num_simulation']
        self.rollout_depth = params['simulation_depth']
        self.keep_subtree = False
        self.keep_tree = False
        self.root = None
        self.tau = params['tau']

    def start(self, observation, info=None):
        '''
        :param observation: numpy array -> (observation shape)
        :return: action : numpy array
        '''
        self.episode_counter += 1
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)

        self.prev_state = self.getStateRepresentation(observation)
        
        if self._uf['network'] is None:
            self.init_uncertainty_network(self.prev_state)
            self.load_uncertainty(self.pre_trained_Unetwork)
            self._uf['batch_size'] = config.U_batch_size
            self._uf['step_size'] = config.U_step_size
            self._uf['training'] = config.U_training

        if self.keep_tree and self.root is None:
            self.root = Node(None, self.prev_state)
            self.expansion(self.root)

        if self.keep_tree:
            self.subtree_node = self.root
        else:
            self.subtree_node = Node(None, self.prev_state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
            # self.render_tree()
        action, sub_tree = self.choose_action()
        action_index = self.getActionIndex(action)

        self.subtree_node = sub_tree
        self.prev_action = torch.tensor([action_index]).unsqueeze(0)
        return action

    def step(self, reward, observation, info=None):
        self.time_step += 1
        self.state = self.getStateRepresentation(observation)
        if not self.keep_subtree:
            self.subtree_node = Node(None, self.state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
            # self.render_tree()
        action, sub_tree = self.choose_action()
        action_index = self.getActionIndex(action)
        self.subtree_node = sub_tree
        self.action = torch.tensor([action_index]).unsqueeze(0)
        reward = torch.tensor([reward], device=self.device)
        if self._uf['training']:
            a = torch.tensor(self.action_list[self.prev_action.item()], device=self.device).unsqueeze(0)
            corrupted_state, _, _, _ = self.model(self.prev_state, a)
            onehot_prev_state = self.get_onehot_state(self.prev_state)
            self.uncertainty_buffer.append(utils.corrupt_transition(onehot_prev_state,
                                                            self.getActionOnehotTorch(self.prev_action),
                                                            self.state,
                                                            corrupted_state))

        self.updateStateRepresentation()
        self.prev_state = self.getStateRepresentation(observation)
        self.prev_action = self.action
        return action

    def end(self, reward):
        reward = torch.tensor([reward], device=self.device)
        if self._uf['training']:
            a = torch.tensor(self.action_list[self.prev_action.item()], device=self.device).unsqueeze(0)

            true_next_state, _, _ = self.true_model(self.prev_state[0], a[0])
            corrupted_state, _, _, _ = self.model(self.prev_state, a)
            onehot_prev_state = self.get_onehot_state(self.prev_state)
            self.uncertainty_buffer.append(utils.corrupt_transition(onehot_prev_state,
                                                            self.getActionOnehotTorch(self.prev_action),
                                                            self.getStateRepresentation(true_next_state),
                                                            corrupted_state))
   
    def init_uncertainty_network(self, state):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''
        nn_state_shape = self.get_onehot_state(state).shape
        self._uf['network'] = UncertaintyNN(nn_state_shape, self.num_actions,
                                                self._uf['layers_type'],
                                                self._uf['layers_features']).to(self.device)
        self.uncerainty_optimizer = optim.Adam(self._uf['network'].parameters(), lr=self._uf['step_size'])

    def choose_action(self):
        max_value = -np.inf
        max_action_list = []
        max_child_list = []
        for child in self.subtree_node.get_childs():
            value = child.num_visits
            if value > max_value:
                max_value = value
                max_action_list = [child.get_action_from_par()]
                max_child_list = [child]
            elif value == max_value:
                max_action_list.append(child.get_action_from_par())
                max_child_list.append(child)
        random_ind = random.randint(0, len(max_action_list) - 1)
        return max_action_list[random_ind], max_child_list[random_ind]

    def train_uncertainty(self):
        loss_sum = 0
        if len(self.uncertainty_buffer) >=  self.minimum_uncertainty_buffer_training:
            for _ in range(self.u_epoch_training // self.episode_counter):
                corrupt_transition_batch = random.sample(self.uncertainty_buffer, k=self._uf['batch_size'])
                loss = self.training_step_uncertainty(corrupt_transition_batch) 
                loss_sum += loss
            print("loss ", self.episode_counter, ", buffer ", len(self.uncertainty_buffer), ":", 
            loss_sum / (self.u_epoch_training))
    
    def training_step_uncertainty(self, corrupt_transition_batch):
        batch = utils.corrupt_transition(*zip(*corrupt_transition_batch))
        true_next_states = torch.cat([s for s in batch.true_state]).float()
        corrupt_next_states = torch.cat([s for s in batch.corrupt_state]).float()
        prev_state_batch = torch.cat(batch.prev_state).float()
        prev_action_batch = torch.cat(batch.prev_action).float()
        predicted_uncertainty = self._uf['network'](prev_state_batch, prev_action_batch)
        true_uncertainty = torch.mean((true_next_states - corrupt_next_states) ** 2, axis=1).unsqueeze(1)
        loss = F.mse_loss(predicted_uncertainty,
                          true_uncertainty)
        self.uncerainty_optimizer.zero_grad()
        loss.backward()
        self.uncerainty_optimizer.step()
        return loss.detach().item()
    
    def get_onehot_state(self, state):
        if self.env == "space_invaders":
            prev_state_pos_onehot = self.getOnehotTorch(torch.tensor([state[0][-6]], dtype=int).unsqueeze(0), 10)
            prev_state_shottimer_onehot = self.getOnehotTorch(torch.tensor([state[0][-1]], dtype=int).unsqueeze(0), 5)
            one_hot_prev_state = torch.cat((state[0][0:-6], state[0][-5:-1])).unsqueeze(0)
            one_hot_prev_state = torch.cat((one_hot_prev_state, prev_state_pos_onehot), dim=1)
            one_hot_prev_state = torch.cat((one_hot_prev_state, prev_state_shottimer_onehot), dim=1)
            
            # one_hot_prev_state = prev_state_pos_onehot
        elif self.env == "freeway":
            state_copy = torch.clone(state)
            one_hot_prev_state = torch.tensor([])
            state_copy[0][3] += 5
            state_copy[0][7] += 5
            state_copy[0][11] += 5
            state_copy[0][15] += 5
            state_copy[0][19] += 5
            state_copy[0][23] += 5
            state_copy[0][27] += 5
            state_copy[0][31] += 5
            for i in range(state_copy.shape[1]):
                if i % 4 == 3:
                    s = self.getOnehotTorch(torch.tensor([state_copy[0][i]], dtype=int).unsqueeze(0), 11)
                else:        
                    s = self.getOnehotTorch(torch.tensor([state_copy[0][i]], dtype=int).unsqueeze(0), 10)
                one_hot_prev_state = torch.cat((one_hot_prev_state, s), dim=1)

        elif self.env == "breakout":
            prev_state_ball_y = self.getOnehotTorch(torch.tensor([state[0][0]], dtype=int).unsqueeze(0), 10)
            prev_state_ball_x = self.getOnehotTorch(torch.tensor([state[0][1]], dtype=int).unsqueeze(0), 10)
            prev_state_ball_dir = self.getOnehotTorch(torch.tensor([state[0][2]], dtype=int).unsqueeze(0), 4)
            prev_state_last_x = self.getOnehotTorch(torch.tensor([state[0][5]], dtype=int).unsqueeze(0), 10)
            prev_state_last_y = self.getOnehotTorch(torch.tensor([state[0][6]], dtype=int).unsqueeze(0), 10)
            prev_state_pos = self.getOnehotTorch(torch.tensor([state[0][3]], dtype=int).unsqueeze(0), 10)

            one_hot_prev_state = torch.cat((state[0][4].unsqueeze(0), state[0][7:])).unsqueeze(0)
            one_hot_prev_state = torch.cat((one_hot_prev_state, prev_state_ball_y), dim=1)
            one_hot_prev_state = torch.cat((one_hot_prev_state, prev_state_ball_x), dim=1)
            one_hot_prev_state = torch.cat((one_hot_prev_state, prev_state_ball_dir), dim=1)
            one_hot_prev_state = torch.cat((one_hot_prev_state, prev_state_pos), dim=1)
            one_hot_prev_state = torch.cat((one_hot_prev_state, prev_state_last_x), dim=1)
            one_hot_prev_state = torch.cat((one_hot_prev_state, prev_state_last_y), dim=1)

        else:
            raise NotImplementedError("agent env name onehot state not implemented")
        return one_hot_prev_state

    def save_uncertainty(self, name):
        with open("MiniAtariResult/Paper/SavedUncertainty/"+name+".p", 'wb') as file:
            pickle.dump(self._uf, file)

    def load_uncertainty(self, name):
        if name is not None:
            with open(name, 'rb') as file:
                self._uf = pickle.load(file)
    
    def save_buffer(self, name):
        with open("MiniAtariResult/Thesis/Buffer/"+name+".p", 'wb') as file:
            pickle.dump(self.uncertainty_buffer, file)
            print("save buffer")
    
    def load_buffer(self, name):
        with open("MiniAtariResult/Thesis/Buffer/"+name+".p", 'rb') as file:
            self.uncertainty_buffer = pickle.load(file)
            print("load buffer")

    def model(self, state, action, calculate_uncertainty=True):
        with torch.no_grad():
            true_next_state, _, reward = self.true_model(state[0], action[0])
            true_next_state = torch.from_numpy(true_next_state).to(self.device)
            corrupted_next_state, is_terminal, _ = self.corrupt_model(state[0], action[0])
            corrupted_next_state = torch.from_numpy(corrupted_next_state).unsqueeze(0)

            corrupted_uncertainty = torch.tensor([0])
            # if want not rounded next_state, replace next_state with _
            if calculate_uncertainty:
                if not self.use_perfect_uncertainty:
                    # # Changing the features ********
                    one_hot_prev_state = self.get_onehot_state(state)
                    # # *********
                    action_index = torch.from_numpy(np.where(self.action_list == [action.item()])[0]).unsqueeze(0)
                    corrupted_uncertainty = self._uf['network'](one_hot_prev_state, self.getActionOnehotTorch(action_index))[0, 0]
                else:
                    true_uncertainty = torch.mean(torch.pow(true_next_state - corrupted_next_state[0], 2).float())
                    corrupted_uncertainty = true_uncertainty

        return corrupted_next_state, is_terminal, reward, corrupted_uncertainty

    def rollout(self, node):
        if ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.rollout_idea == 1:
            sum_returns = 0
            for _ in range(self.num_rollouts):
                depth = 0
                is_terminal = node.is_terminal
                state = node.get_state()

                gamma_prod = 1
                single_return = 0
                sum_uncertainty = 0
                return_list = []
                weight_list = []
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    next_state, is_terminal, reward, uncertainty = self.model(state, action)
                    uncertainty = uncertainty.item()
                    single_return += reward * gamma_prod
                    gamma_prod *= self.gamma
                    sum_uncertainty += uncertainty

                    return_list.append(single_return)
                    weight_list.append(sum_uncertainty)
                    depth += 1
                    state = next_state
                return_list = np.asarray(return_list)
                weight_list = np.asarray(weight_list)

                weights = np.exp(-weight_list / self.tau) / np.sum(np.exp(-weight_list / self.tau))
                # print(return_list)
                # print(weight_list)
                # print(weights)
                # print(np.average(return_list, weights=weights), np.average(return_list))
                if len(weights) > 0:
                    uncertain_return = np.average(return_list, weights=weights)
                else:  # the starting node is a terminal state
                    uncertain_return = 0
                sum_returns += uncertain_return
            return sum_returns / self.num_rollouts

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.rollout_idea == 3:
            sum_returns = 0
            for i in range(self.num_rollouts):
                depth = 0
                single_return = 0
                is_terminal = node.is_terminal
                state = node.get_state()
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    next_state, is_terminal, reward, uncertainty = self.model(state, action)
                    uncertainty = uncertainty.item()
                    single_return += reward * uncertainty
                    depth += 1
                    state = next_state

                sum_returns += single_return
            return sum_returns / self.num_rollouts

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.rollout_idea == 4:
            # with torch.no_grad():
            #     bootstrap_value = torch.mean(self.value_function['q']['network'](node.get_state())).item()
            # return bootstrap_value

            sum_returns = 0
            for i in range(self.num_rollouts):
                depth = 0
                single_return = 0
                is_terminal = node.is_terminal
                state = node.get_state()
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    next_state, is_terminal, reward, _ = self.model(state, action, calculate_uncertainty=False)
                    single_return += reward
                    depth += 1
                    state = next_state
                if not is_terminal:
                    with torch.no_grad():
                        bootstrap_value = torch.mean(self.value_function['q']['network'](state)).item() 
                    single_return += bootstrap_value

                sum_returns += single_return
            return sum_returns / self.num_rollouts

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.rollout_idea == 5:
            uncertainty_list = []
            return_list = []
            for i in range(self.num_rollouts):
                depth = 0
                single_return = 0
                sum_uncertainty = 0
                gamma_prod = 1
                is_terminal = node.is_terminal
                state = node.get_state()
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    next_state, is_terminal, reward, uncertainty = self.model(state, action)
                    sum_uncertainty += gamma_prod * uncertainty
                    gamma_prod *= self.gamma
                    single_return += reward
                    depth += 1
                    state = next_state
                uncertainty_list.append(sum_uncertainty)
                return_list.append(single_return)

            uncertainty_list = np.asarray(uncertainty_list)
            softmax_uncertainty_list = np.exp(-uncertainty_list / self.tau) / np.sum(np.exp(-uncertainty_list / self.tau))
            weighted_avg = np.average(return_list, weights=softmax_uncertainty_list)
            return weighted_avg

        else:
            sum_returns = 0
            for i in range(self.num_rollouts):
                rollout_path = []
                depth = 0
                single_return = 0
                is_terminal = node.is_terminal
                state = node.get_state()
                while not is_terminal and depth < self.rollout_depth:
                    action_index = torch.randint(0, self.num_actions, (1, 1))
                    action = torch.tensor(self.action_list[action_index]).unsqueeze(0)
                    try:
                        next_state, is_terminal, reward, _ = self.model(state, action, calculate_uncertainty=False)
                        rollout_path.append([state, next_state, is_terminal])
                    except:
                        for i in rollout_path:
                            with open("log.txt", "a") as file:
                                file.write("rolloutpath_state:"+str(i[0]))
                                file.write("rolloutpath_nextstate:"+ str(i[1]))
                                file.write("rolloutpath_terminal:"+str(i[2]))
                                file.write("____________")
                        exit(0)
                    single_return += reward
                    depth += 1
                    state = next_state

                sum_returns += single_return
            return sum_returns / self.num_rollouts

    def selection(self):
        if ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.backpropagate_idea == 1:
            selected_node = self.subtree_node
            while len(selected_node.get_childs()) > 0:
                max_uct_value = -np.inf
                child_values = list(
                    map(lambda n: n.get_weighted_avg_value() + n.reward_from_par, selected_node.get_childs()))
                max_child_value = max(child_values)
                min_child_value = min(child_values)

                child_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, selected_node.get_childs())))
                softmax_denominator = np.sum(np.exp(child_uncertainties / self.tau))
                softmax_uncertainties = np.exp(child_uncertainties / self.tau) / softmax_denominator

                for ind, child in enumerate(selected_node.get_childs()):
                    if child.num_visits == 0:
                        selected_node = child
                        break
                    else:

                        child_value = child_values[ind]
                        # child_value = child.get_avg_value() + child.reward_from_par
                        child_uncertainty = child_uncertainties[ind]
                        softmax_uncertainty = softmax_uncertainties[ind]

                        if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                            child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
                        elif min_child_value == max_child_value:
                            child_value = 0.5
                        if ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.selection_idea == 1:
                            uct_value = (child_value + \
                                     self.C * ((np.log(child.parent.num_visits) / child.num_visits) ** 0.5)) *\
                                    (1 - softmax_uncertainty)
                        else:
                            uct_value = (child_value + \
                                        self.C * ((np.log(child.parent.num_visits) / child.num_visits) ** 0.5))
                        # print("old:", uct_value - child_uncertainty, "  new:",uct_value, "  unc:", child_uncertainty)
                    if max_uct_value < uct_value:
                        max_uct_value = uct_value
                        selected_node = child
            return selected_node

        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.selection_idea == 1:
            selected_node = self.subtree_node
            while len(selected_node.get_childs()) > 0:
                max_uct_value = -np.inf
                child_values = list(map(lambda n: n.get_avg_value() + n.reward_from_par, selected_node.get_childs()))
                max_child_value = max(child_values)
                min_child_value = min(child_values)

                child_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, selected_node.get_childs())))
                softmax_denominator = np.sum(np.exp(child_uncertainties / self.tau))
                softmax_uncertainties = np.exp(child_uncertainties / self.tau)/softmax_denominator

                for ind, child in enumerate(selected_node.get_childs()):
                    if child.num_visits == 0:
                        selected_node = child
                        break
                    else:

                        child_value = child_values[ind]
                        child_value = child.get_avg_value() + child.reward_from_par
                        child_uncertainty = child_uncertainties[ind]
                        softmax_uncertainty = softmax_uncertainties[ind]

                        if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                            child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
                        elif min_child_value == max_child_value:
                            child_value = 0.5

                        uct_value = (child_value + \
                                     self.C * ((np.log(child.parent.num_visits) / child.num_visits) ** 0.5)) *\
                                    (1 - softmax_uncertainty)
                        # print("old:", uct_value - child_uncertainty, "  new:",uct_value, "  unc:", child_uncertainty)
                    if max_uct_value < uct_value:
                        max_uct_value = uct_value
                        selected_node = child
            return selected_node

        else:
            return MCTSAgent.selection(self)

    def expansion(self, node):
        if ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.expansion_idea == 1:
            for a in self.action_list:
                action = torch.tensor(a).unsqueeze(0)
                next_state, is_terminal, reward, uncertainty = self.model(node.get_state(),
                                                                          action, calculate_uncertainty=False)  # with the assumption of deterministic model
                # if np.array_equal(next_state, node.get_state()):
                #     continue
                with torch.no_grad():
                    value = torch.mean(self.value_function['q']['network'](next_state)).item() 

                child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                             value=value, uncertainty=uncertainty.item())
                node.add_child(child)
        
        elif ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.expansion_idea == 2:
            uncertainty_list = []
            possible_children = []
            for a in self.action_list:
                action = torch.tensor(a).unsqueeze(0)
                next_state, is_terminal, reward, uncertainty = self.model(node.get_state(),
                                                                          action)  # with the assumption of deterministic model
                # if np.array_equal(next_state, node.get_state()):
                #     continue
                value = self.get_initial_value(next_state)
                child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                             value=value, uncertainty=uncertainty.item())
                uncertainty_list.append(uncertainty)
                possible_children.append(child)
            uncertainty_list = np.asarray(uncertainty_list)
            norm_uncertainties = uncertainty_list / np.sum(uncertainty_list)
            excluded_child = None
            if np.random.rand() < (1 - self.tau) and np.sum(uncertainty_list) != 0:
                excluded_child = np.random.choice(len(possible_children), p=norm_uncertainties)
            for i, child in enumerate(possible_children):
                if i != excluded_child:
                    node.add_child(child)
        
        else:
            for a in self.action_list:
                action = torch.tensor(a).unsqueeze(0)
                next_state, is_terminal, reward, uncertainty = self.model(node.get_state(),
                                                                          action)  # with the assumption of deterministic model
                # if np.array_equal(next_state, node.get_state()):
                #     continue
                value = self.get_initial_value(next_state)
                child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                             value=value, uncertainty=uncertainty.item())
                node.add_child(child)

    def backpropagate(self, node, value):
        #did not implement tau for backpropagation ***************
        if ImperfectMCTSAgentUncertaintyHandDesignedModelValueFunction.backpropagate_idea == 1:
            while node is not None:
                node.add_to_values(value)
                node.inc_visits()
                if node.parent is not None:
                    siblings = node.parent.get_childs()
                    siblings_uncertainties = np.asarray(list(map(lambda n: n.uncertainty, siblings)))
                    softmax_denominator = np.sum(np.exp(-siblings_uncertainties / self.tau))
                    value *= self.gamma
                    value += node.reward_from_par
                    value *= np.exp(-node.uncertainty / self.tau) / softmax_denominator
                node = node.parent
        else:
            while node is not None:
                node.add_to_values(value)
                node.inc_visits()
                value *= self.gamma
                value += node.reward_from_par
                node = node.parent

    def getOnehotTorch(self, index, num_actions):
        '''
        action = index torch
        output = onehot torch
        '''
        batch_size = index.shape[0]
        onehot = torch.zeros([batch_size, num_actions], device="cpu")
        onehot.scatter_(1, index, 1)
        return onehot