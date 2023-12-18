import numpy as np
import torch
import random

from Agents.BaseAgent import BaseAgent
from DataStructures.Node import Node as Node
from Networks.RepresentationNN.StateRepresentation import StateRepresentation

class MCTSAgent(BaseAgent):
    name = "MCTSAgent"

    def __init__(self, params={}):

        self.time_step = 0

        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]

        self.gamma = params['gamma']
        self.epsilon = params['epsilon']

        self.device = params['device']


        self.true_model = params['true_fw_model']

        # MCTS parameters
        self.C = params['c']
        self.num_iterations = params['num_iteration']
        self.num_rollouts = params['num_simulation']
        self.rollout_depth = params['simulation_depth']
        self.keep_subtree = False
        self.keep_tree = False
        self.root = None

        self.is_model_imperfect = False
        self.corrupt_prob = 0.025
        self.corrupt_step = 1

        self._sr = dict(network=None,
                        layers_type=[],
                        layers_features=[],
                        batch_size=None,
                        step_size=None,
                        batch_counter=None,
                        training=False)
    
    def start(self, observation, info=None):
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)
        
        state = self.getStateRepresentation(observation)
        
        if self.keep_tree and self.root is None:
            self.root = Node(None, state)
            self.expansion(self.root)

        if self.keep_tree:
            self.subtree_node = self.root
        else:
            self.subtree_node = Node(None, state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree
        return action

    def step(self, reward, observation):

        state = self.getStateRepresentation(observation)
        if not self.keep_subtree:
            self.subtree_node = Node(None, state)
            self.expansion(self.subtree_node)

        for i in range(self.num_iterations):
            self.MCTS_iteration()
        action, sub_tree = self.choose_action()
        self.subtree_node = sub_tree
        return action

    def end(self, reward):
        pass

    def get_initial_value(self, state):
        return 0

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

    def MCTS_iteration(self):
        selected_node = self.selection()
        if selected_node.is_terminal:
            self.backpropagate(selected_node, 0)
        elif selected_node.num_visits == 0:
            rollout_value = self.rollout(selected_node)
            self.backpropagate(selected_node, rollout_value)
        else:
            self.expansion(selected_node)
            rollout_value = self.rollout(selected_node.get_childs()[0])
            self.backpropagate(selected_node.get_childs()[0], rollout_value)

    def selection(self):
        selected_node = self.subtree_node
        while len(selected_node.get_childs()) > 0:
            max_uct_value = -np.inf
            child_values = list(map(lambda n: n.get_avg_value()+n.reward_from_par, selected_node.get_childs()))
            max_child_value = max(child_values)
            min_child_value = min(child_values)
            for ind, child in enumerate(selected_node.get_childs()):
                if child.num_visits == 0:
                    selected_node = child
                    break
                else:
                    child_value = child_values[ind]
                    if min_child_value != np.inf and max_child_value != np.inf and min_child_value != max_child_value:
                        child_value = (child_value - min_child_value) / (max_child_value - min_child_value)
                    uct_value = child_value + \
                                self.C * ((np.log(child.parent.num_visits) / child.num_visits) ** 0.5)
                if max_uct_value < uct_value:
                    max_uct_value = uct_value
                    selected_node = child
        return selected_node

    def expansion(self, node):
        for a in range(self.num_actions):
            action_index = torch.tensor([a]).unsqueeze(0)
            next_state, is_terminal, reward, _ = self.model(node.get_state(),
                                                              action_index)
            value = self.get_initial_value(next_state)
            child = Node(node, next_state, is_terminal=is_terminal, action_from_par=a, reward_from_par=reward,
                         value=value)
            node.add_child(child)

    def rollout(self, node):
        sum_returns = 0
        for i in range(self.num_rollouts):
            depth = 0
            single_return = 0
            is_terminal = node.is_terminal
            state = node.get_state()
            while not is_terminal and depth < self.rollout_depth:
                action_index = torch.randint(0, self.num_actions, (1, 1))
                next_state, is_terminal, reward, _ = self.model(state, action_index)
                single_return += reward
                depth += 1
                state = next_state

            sum_returns += single_return
        return sum_returns / self.num_rollouts

    def backpropagate(self, node, value):
        while node is not None:
            node.add_to_values(value)
            node.inc_visits()
            value *= self.gamma
            value += node.reward_from_par
            node = node.parent

    def model(self, state, action_index):
        state_np = state.cpu().numpy()
        action_index = action_index.cpu().numpy()
        next_state_np, is_terminal, reward = self.true_model(state_np[0], action_index.item())
        next_state = torch.from_numpy(next_state_np).unsqueeze(0).to(self.device)
        return next_state, is_terminal, reward, 0

    def getActionIndex(self, action):
        for i, a in enumerate(self.action_list):
            if np.array_equal(a, action):
                return i
        raise ValueError("action is not defined")

    def updateStateRepresentation(self):
        if len(self._sr['layers_type']) == 0:
            return None
        if self._sr['batch_counter'] == self._sr['batch_size'] and self._sr['training']:
            self.updateNetworkWeights(self._sr['network'], self._sr['step_size'] / self._sr['batch_size'])
            self._sr['batch_counter'] = 0

    def init_s_representation_network(self, observation):
        '''
        :param observation: numpy array
        :return: None
        '''
        nn_state_shape = observation.shape
        self._sr['network'] = StateRepresentation(nn_state_shape,
                                                  self._sr['layers_type'],
                                                  self._sr['layers_features']).to(self.device)

    def getStateRepresentation(self, observation, gradient=False):
        '''
        :param observation: numpy array -> [obs_shape]
        :param gradient: boolean
        :return: torch including batch -> [1, state_shape]
        '''
        if gradient:
            self._sr['batch_counter'] += 1
        observation = torch.tensor([observation], device=self.device)
        if gradient:
            rep = self._sr['network'](observation)
        else:
            with torch.no_grad():
                rep = self._sr['network'](observation)
        return rep