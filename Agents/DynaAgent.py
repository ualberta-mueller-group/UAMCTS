import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

import utils
from Agents.BaseAgent import BaseAgent
from Networks.ValueFunctionNN.StateActionValueFunction import StateActionVFNN
from Networks.ValueFunctionNN.StateValueFunction import StateVFNN
from Networks.RepresentationNN.StateRepresentation import StateRepresentation
from Networks.ModelNN.StateTransitionModel import StateTransitionModel
from Networks.ModelNN.StateTransitionModel import StateTransitionModelHeter
import pickle


# this is an DQN agent.
class DynaAgent(BaseAgent):
    name = 'DynaAgent'

    def __init__(self, params={}):
        self.model_loss = []
        self.time_step = 0
        self.writer_iterations = 0
        self.prev_state = None
        self.state = None

        self.action_list = params['action_list']
        self.num_actions = self.action_list.shape[0]
        self.actions_shape = self.action_list.shape[1:]

        self.gamma = params['gamma']
        self.epsilon = params['epsilon_min']

        self.transition_buffer = []
        self.transition_buffer_size = 2 ** 12

        self.policy_values = 'q'  # 'q' or 's' or 'qs'
        # self.policy_values = params['vf']['type']  # 'q' or 's' or 'qs'

        # self._vf = {'q': dict(network=None,
        #                       layers_type=params['vf']['layers_type'],
        #                       layers_features=params['vf']['layers_features'],
        #                       action_layer_num=params['vf']['action_layer_num'],
        #                       # if one more than layer numbers => we will have num of actions output
        #                       batch_size=16,
        #                       step_size=params['max_stepsize'],
        #                       training=True),
        #             's': dict(network=None,
        #                       layers_type=params['vf']['layers_type'],
        #                       layers_features=params['vf']['layers_features'],
        #                       batch_size=16,
        #                       step_size=params['max_stepsize'],
        #                       training=False)}
        self.loadValueFunction(params['vf'])
        self._sr = dict(network=None,
                        layers_type=[],
                        layers_features=[],
                        batch_size=None,
                        step_size=None,
                        batch_counter=None,
                        training=False)

        self._target_vf = {'q':dict(network=None,
                               counter=0,
                               layers_num=None,
                               action_layer_num=None,
                               update_rate=500),
                           's':dict(network=None,
                               counter=0,
                               update_rate=500)}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #params['device']

        self.num_steps = 0
        self.num_terminal_steps = 0

        self.is_pretrained = True

    def start(self, observation, info=None):
        '''
        :param observation: numpy array -> (observation shape)
        :return: action : numpy array
        '''
        if self._sr['network'] is None:
            self.init_s_representation_network(observation)

        self.prev_state = self.getStateRepresentation(observation)
        if self._vf['q']['network'] is None and self._vf['q']['training']:
            print("init VF !")
            self.init_q_value_function_network(self.prev_state)  # a general state action VF for all actions

        # if self._vf['s']['network'] is None and self._vf['s']['training']:
        #     self.init_s_value_function_network(self.prev_state)  # a separate state VF for each action
        
        self.setTargetValueFunction()
        self.prev_action = self.policy(self.prev_state)
        return self.action_list[self.prev_action.cpu().item()]

    def step(self, reward, observation, info=None):
        self.time_step += 1

        self.state = self.getStateRepresentation(observation)

        reward = torch.tensor([reward], device=self.device)
        self.action = self.policy(self.state)

        # store the new transition in buffer

        self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                     self.prev_action,
                                                     reward,
                                                     self.state,
                                                     self.action, False, self.time_step, 0))
        # update target
        if self._target_vf['q']['counter'] >= self._target_vf['q']['update_rate']:
            self.setTargetValueFunction()

        # update value function with the buffer
        if self._vf['q']['training']:
            print("update VF !")
            if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')
        # if self._vf['s']['training']:
        #     if len(self.transition_buffer) >= self._vf['s']['batch_size']:
        #         transition_batch = self.getTransitionFromBuffer(n=self._vf['s']['batch_size'])
        #         self.updateValueFunction(transition_batch, 's')

        self.updateStateRepresentation()

        self.prev_state = self.getStateRepresentation(observation)
        self.prev_action = self.action

        return self.action_list[self.prev_action.cpu().item()]

    def end(self, reward):
        reward = torch.tensor([reward], device=self.device)

        self.updateTransitionBuffer(utils.transition(self.prev_state,
                                                     self.prev_action,
                                                     reward,
                                                     None,
                                                     None, True, self.time_step, 0))

        if self._vf['q']['training']:
            print("Update VF!")
            if len(self.transition_buffer) >= self._vf['q']['batch_size']:
                transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
                self.updateValueFunction(transition_batch, 'q')
        # if self._vf['s']['training']:
        #     if len(self.transition_buffer) >= self._vf['s']['batch_size']:
        #         transition_batch = self.getTransitionFromBuffer(n=self._vf['q']['batch_size'])
        #         self.updateValueFunction(transition_batch, 's')

        self.updateStateRepresentation()

    def policy(self, state):
        '''
        :param state: torch -> (1, state_shape)
        :return: action: index torch
        '''
        if random.random() < self.epsilon:
            ind = torch.tensor([[random.randrange(self.num_actions)]],
                               device=self.device, dtype=torch.long)
            return ind
        with torch.no_grad():
            v = []
            if self.policy_values == 'q':
                ind = self._vf['q']['network'](state).max(1)[1].view(1, 1)
                return ind
            else:
                raise ValueError('policy is not defined')

    # ***
    def init_q_value_function_network(self, state):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''
        nn_state_shape = state.shape
        self._vf['q']['network'] = StateActionVFNN(nn_state_shape, self.num_actions,
                                                   self._vf['q']['layers_type'],
                                                   self._vf['q']['layers_features'],
                                                   self._vf['q']['action_layer_num']).to(self.device)
        self._vf['q']['network'] = StateActionVFNN(nn_state_shape, self.num_actions).to(self.device)
        # if self.is_pretrained:
        #     value_function_file = ""
        #     self.loadValueFunction(value_function_file)
        #     self._vf['q']['training'] = False
        self.optimizer = optim.Adam(self._vf['q']['network'].parameters(), lr=self._vf['q']['step_size'])

    def init_s_value_function_network(self, state):
        '''
        :param state: torch -> (1, state)
        :return: None
        '''
        nn_state_shape = state.shape
        self._vf['s']['network'] = StateVFNN(nn_state_shape,
                                             self._vf['s']['layers_type'],
                                             self._vf['s']['layers_features']).to(self.device)
        self.optimizer = optim.Adam(self._vf['s']['network'].parameters(), lr=self._vf['s']['step_size'])

    def init_s_representation_network(self, observation):
        '''
        :param observation: numpy array
        :return: None
        '''
        nn_state_shape = observation.shape
        self._sr['network'] = StateRepresentation(nn_state_shape,
                                                  self._sr['layers_type'],
                                                  self._sr['layers_features']).to(self.device)

    def updateValueFunction(self, transition_batch, vf_type):

        batch = utils.transition(*zip(*transition_batch))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None,
                      batch.state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.state
                                           if s is not None])
        prev_state_batch = torch.cat(batch.prev_state)
        prev_action_batch = torch.cat(batch.prev_action)
        reward_batch = torch.cat(batch.reward)

        # BEGIN DQN
        state_action_values = self._vf['q']['network'](prev_state_batch).gather(1, prev_action_batch)
        next_state_values = torch.zeros(self._vf['q']['batch_size'], device=self.device)
        next_state_values[non_final_mask] = self._target_vf['q']['network'](non_final_next_states).max(1)[0].detach()
        # END DQN

        # BEGIN SARSA
        # non_final_next_actions = torch.cat([a for a in batch.action
        #                                    if a is not None])
        # state_action_values = self._vf['q']['network'](prev_state_batch).gather(1, prev_action_batch)
        # next_state_values = torch.zeros(self._vf['q']['batch_size'], device=self.device)
        # next_state_values[non_final_mask] = self._target_vf['network'](non_final_next_states).gather(1, non_final_next_actions).detach()[:, 0]
        # END SARSA

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        loss = F.mse_loss(state_action_values,
                          expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._target_vf['q']['counter'] += 1

    def getStateRepresentation(self, observation, gradient=False):
        '''
        :param observation: numpy array -> [obs_shape]
        :param gradient: boolean
        :return: torch including batch -> [1, state_shape]
        '''
        if gradient:
            self._sr['batch_counter'] += 1
        observation = torch.tensor(observation, device=self.device).unsqueeze(0)
        if gradient:
            rep = self._sr['network'](observation)
        else:
            with torch.no_grad():
                rep = self._sr['network'](observation)
        return rep

    def updateStateRepresentation(self):

        if len(self._sr['layers_type']) == 0:
            return None
        if self._sr['batch_counter'] == self._sr['batch_size'] and self._sr['training']:
            self.updateNetworkWeights(self._sr['network'], self._sr['step_size'] / self._sr['batch_size'])
            self._sr['batch_counter'] = 0

    def setTargetValueFunction(self):
        nn_state_shape = self.prev_state.shape
        if self._target_vf['q']['network'] is None:    
            self._target_vf['q']['network'] = StateActionVFNN(
            nn_state_shape,
            self.num_actions,
            self._vf['q']['layers_type'],
            self._vf['q']['layers_features'],
            self._vf['q']['action_layer_num']).to(self.device)

        # if self._target_vf['s']['network'] is None:
        #     self._target_vf['s']['network'] = StateVFNN(nn_state_shape,
        #                                      self._vf['s']['layers_type'],
        #                                      self._vf['s']['layers_features']).to(self.device)


        self._target_vf['q']['network'].load_state_dict(self._vf['q']['network'].state_dict())  # copy weights and stuff
        # self._target_vf['s']['network'].load_state_dict(self._vf['s']['network'].state_dict())  # copy weights and stuff

        self._target_vf['q']['action_layer_num'] = self._vf['q']['action_layer_num']
        self._target_vf['layers_num'] = len(self._vf['q']['layers_type'])
       
        self._target_vf['q']['counter'] = 0
        # self._target_vf['s']['counter'] = 0

    def getTransitionFromBuffer(self, n):
        # both model and value function are using this buffer
        if len(self.transition_buffer) < n:
            n = len(self.transition_buffer)
        return random.sample(self.transition_buffer, k=n)

    def updateTransitionBuffer(self, transition):
        self.num_steps += 1
        if transition.is_terminal:
            self.num_terminal_steps += 1
        self.transition_buffer.append(transition)
        if len(self.transition_buffer) > self.transition_buffer_size:
            self.removeFromTransitionBuffer()

    def removeFromTransitionBuffer(self):
        self.num_steps -= 1
        transition = self.transition_buffer.pop(0)
        if transition.is_terminal:
            self.num_terminal_steps -= 1

    def getActionIndex(self, action):
        for i, a in enumerate(self.action_list):
            if np.array_equal(a, action):
                return i
        raise ValueError("action is not defined")

    def getActionOnehot(self, action):
        res = np.zeros([len(self.action_list)])
        res[self.getActionIndex(action)] = 1
        return res

    def getActionOnehotTorch(self, action):
        '''
        action = index torch
        output = onehot torch
        '''
        batch_size = action.shape[0]
        num_actions = len(self.action_list)
        onehot = torch.zeros([batch_size, num_actions], device=self.device)
        onehot.scatter_(1, action, 1)
        return onehot

    def saveValueFunction(self, name):
        with open(name, "wb") as file:
            pickle.dump(self._vf, file)

    def loadValueFunction(self, name):
        with open(name, "rb") as file:
            self._vf = pickle.load(file)