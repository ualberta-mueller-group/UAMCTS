import pickle
import random
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

import utils, config
from Agents.MCTSAgent import MCTSAgent as MCTSAgent
from Agents.DynaAgent import DynaAgent
from DataStructures.Node import Node as Node
from Networks.ModelNN.StateTransitionModelTwoWay import StateTransitionModel


class MCTSAgentTwoWayOnlineModel(DynaAgent, MCTSAgent):
    name = "MCTSAgentTwoWayOnlineModel"
    
    def __init__(self, params={}):
        self.env = params['env_name']
        self.device = params['device']

        self.episode_counter = 0
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

        self.minimum_transition_buffer_training = config.minimum_transition_buffer_training
        self.st_training_steps = [config.st_epoch_training_rate * i for i in range(1, config.num_episode * config.max_step_each_episode // config.st_epoch_training_rate)]
        self.st_epoch_training = config.st_epoch_training

        self._st = {'network':None,
                    'batch_size': config.st_batch_size,
                    'step_size':config.st_step_size,
                    'layers_type':config.st_layers_type,
                    'layers_features':config.st_layers_features,
                    'training':config.st_training}

        self._sr = dict(network=None,
                        layers_type=[],
                        layers_features=[],
                        batch_size=None,
                        step_size=None,
                        batch_counter=None,
                        training=False)

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

        self.transition_has_been_trained = False 

    def start(self, observation, info=None):
        '''
        :param observation: numpy array -> (observation shape)
        :return: action : numpy array
        '''
        self.episode_counter += 1

        if self._sr['network'] is None:
            self.init_s_representation_network(observation)

        self.prev_state = self.getStateRepresentation(observation)
        
        if self._st['network'] is None:
            self.init_transition_network(self.prev_state)

        if self.keep_tree and self.root is None:
            self.root = Node(None, self.prev_state, num_visits=1)
            self.expansion(self.root)

        if self.keep_tree:
            self.subtree_node = self.root
        else:
            self.subtree_node = Node(None, self.prev_state, num_visits=1)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        action_index = self.getActionIndex(action)
        self.subtree_node = sub_tree
        self.prev_action = torch.tensor([action_index]).unsqueeze(0)

        return action

    def step(self, reward, observation, info=None):
        self.time_step += 1
        self.state = self.getStateRepresentation(observation)
        if not self.keep_subtree:
            self.subtree_node = Node(None, self.state, num_visits=1)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        action_index = self.getActionIndex(action)
        self.subtree_node = sub_tree
        self.action = torch.tensor([action_index]).unsqueeze(0)
        reward = torch.tensor([reward], device=self.device)
        if self._st['training']:
            a = torch.tensor(self.action_list[self.prev_action.item()], device=self.device).unsqueeze(0)
            corrupted_state, _, _, _ = self.model(self.prev_state, a)
            onehot_prev_state = self.get_onehot_state(self.prev_state)
            self.transition_buffer.append(utils.corrupt_transition(onehot_prev_state,
                                                            self.prev_action,
                                                            self.state,
                                                            corrupted_state))
                                                            
        if self._st['training'] and self.time_step in self.st_training_steps:
            self.train_transition()
           
        self.updateStateRepresentation()
        self.prev_state = self.getStateRepresentation(observation)
        self.prev_action = self.action
        return action

    def end(self, reward):
        self.time_step += 1
        reward = torch.tensor([reward], device=self.device)
        if self._st['training']:
            a = torch.tensor(self.action_list[self.prev_action.item()], device=self.device).unsqueeze(0)
            true_next_state, _, _ = self.true_model(self.prev_state[0], a[0])
            corrupted_state, _, _, _ = self.model(self.prev_state, a)
            onehot_prev_state = self.get_onehot_state(self.prev_state)
            for i in range(10):
                self.transition_buffer.append(utils.corrupt_transition(onehot_prev_state,
                                                            self.prev_action,
                                                            self.getStateRepresentation(true_next_state),
                                                            corrupted_state))
                             
        
        if self._st['training'] and self.time_step in self.st_training_steps:
            self.train_transition()
   
    def init_transition_network(self, state):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''
        nn_state_shape = self.get_onehot_state(state).shape
        self._st['network'] = StateTransitionModel(nn_state_shape, self.num_actions,
                                                self._st['layers_type'],
                                                self._st['layers_features']).to(self.device)
        self.transition_optimizer = optim.Adam(self._st['network'].parameters(), lr=self._st['step_size'])

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

    def train_transition(self):
        if len(self.transition_buffer) < max(self.minimum_transition_buffer_training, self._st['batch_size']):
            return
        loss_sum = 0
        for _ in range(self.st_epoch_training):
            corrupt_transition_batch = random.sample(self.transition_buffer, k=self._st['batch_size'])
            loss = self.training_step_transition(corrupt_transition_batch) 
            loss_sum += loss
        print("loss ", self.episode_counter, ", buffer ", len(self.transition_buffer), ":", 
        loss_sum / (self.st_epoch_training))
        self.transition_has_been_trained = True

    def training_step_transition(self, corrupt_transition_batch):
        batch = utils.corrupt_transition(*zip(*corrupt_transition_batch))
        true_next_states = torch.cat([s for s in batch.true_state]).float()
        prev_state_batch = torch.cat(batch.prev_state).float()
        prev_action_batch = torch.cat(batch.prev_action).float()
        predicted_next_state = self._st['network'](prev_state_batch, prev_action_batch)
        total_loss = F.mse_loss(predicted_next_state, true_next_states)
        self.transition_optimizer.zero_grad()
        total_loss.backward()
        self.transition_optimizer.step()
        return total_loss.detach().item()
    
    def get_onehot_state(self, state):
        if self.env == "two_way":
            state_copy = torch.clone(state)
            one_hot_prev_state = state_copy
        else:
            raise NotImplementedError("agent env name onehot state not implemented")
        return one_hot_prev_state

    def model(self, state, action, calculate_uncertainty=True):
        if not self.transition_has_been_trained:
            corrupted_next_state, is_terminal, reward = self.corrupt_model(state[0], action[0])
            corrupted_next_state = torch.from_numpy(np.array([corrupted_next_state]))
            return corrupted_next_state, is_terminal, reward, 0

        with torch.no_grad():
            one_hot_prev_state = self.get_onehot_state(state)
            one_hot_prev_state = one_hot_prev_state.unsqueeze(0)
            if self.env == "two_way":
                action = action.detach().numpy()[0]
                for i in range(len(self.action_list)):
                    if np.array_equal(action, self.action_list[i]):
                        ind = i
                action_index = torch.tensor([ind]).unsqueeze(0)
            else:
                action_index = torch.from_numpy(np.where(self.action_list == [action.item()])[0]).unsqueeze(0)
            predicted_next_state = self._st['network'](one_hot_prev_state, action_index)
        is_terminal = False
        reward = 0
        self.l = torch.tensor([[0.0, 0.0]])
        self.u = torch.tensor([[2.0, 6.0]])
        predicted_next_state = torch.max(torch.min(predicted_next_state, self.u), self.l)
        predicted_next_state = torch.round(predicted_next_state)
        if predicted_next_state[0][0] == 1 and predicted_next_state[0][1] == 6:
            is_terminal = True
            reward = 10
        return predicted_next_state, is_terminal, reward, 0

    def rollout(self, node):
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
                next_state, is_terminal, reward, _ = self.model(state, action, calculate_uncertainty=False)
                rollout_path.append([state, next_state, is_terminal])
                single_return += reward
                depth += 1
                state = next_state
            sum_returns += single_return
        return sum_returns / self.num_rollouts

    def selection(self):
        return MCTSAgent.selection(self)

    def expansion(self, node):
        for a in self.action_list:
            action = torch.tensor(a).unsqueeze(0)
            next_state, is_terminal, reward, uncertainty = self.model(node.get_state(),
                                                                        action)
            value = self.get_initial_value(next_state)
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                            value=value, uncertainty=0)
            node.add_child(child)

    def backpropagate(self, node, value):
        while node is not None:
            node.inc_visits()
            node.add_to_values(value)
            value *= self.gamma
            value += node.reward_from_par
            node = node.parent

    def getOnehotTorch(self, index, num_actions):
        '''
        action = index torch
        output = onehot torch
        '''
        batch_size = index.shape[0]
        onehot = torch.zeros([batch_size, num_actions], device=self.device)
        onehot.scatter_(1, index, 1)
        return onehot