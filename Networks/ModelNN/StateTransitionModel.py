import torch
import torch.nn as nn
import torch.nn.functional as F


class StateTransitionModel(nn.Module):
    def __init__(self, state_shape, action_shape, layers_type, layers_features):
        # state : B, state_size(linear)
        # action: A
        super(StateTransitionModel, self).__init__()
        self.layers_type = layers_type
        self.layers = []
        state_size = state_shape[1]
        action_size = action_shape[1]

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    linear_input_size = state_size + action_size
                    layer = nn.Linear(linear_input_size, layers_features[i])
                else:
                    layer = nn.Linear(layers_features[i - 1] + action_size, layers_features[i])
                self.add_module('hidden_layer_' + str(i), layer)
                self.layers.append(layer)
            else:
                raise ValueError("layer is not defined")

        self.head = nn.Linear(layers_features[-1], state_size)


    def forward(self, state, action):
        x = None
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim= 1)
                a = action.flatten(start_dim=1)
                x = torch.cat((x.float(), a.float()), dim=1)
                x = self.layers[i](x.float())
                x = torch.relu(x)
            else:
                raise ValueError("layer is not defined")
        head = self.head(x.float())
        return head

class StateTransitionModelHeter(nn.Module):
    def __init__(self, state_shape, action_shape, layers_type, layers_features):
        # state : B, state_size(linear)
        # action: A
        super(StateTransitionModelHeter, self).__init__()
        self.layers_type = layers_type
        self.mu_layers = []
        self.var_layers = []
        state_size = state_shape[1]
        action_size = action_shape[1]

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    linear_input_size = state_size + action_size
                    mu_layer = nn.Linear(linear_input_size, layers_features[i])
                    var_layer = nn.Linear(linear_input_size, layers_features[i])
                else:
                    mu_layer = nn.Linear(layers_features[i - 1] + action_size, layers_features[i])
                    var_layer = nn.Linear(layers_features[i - 1] + action_size, layers_features[i])
                self.add_module('hidden_layer_' + str(i), mu_layer)
                self.add_module('hidden_varlayer_' + str(i), var_layer)
                self.mu_layers.append(mu_layer)
                self.var_layers.append(var_layer)
            else:
                raise ValueError("layer is not defined")

        self.mu_head = nn.Linear(layers_features[-1], state_size)
        self.var_head = nn.Linear(layers_features[-1], state_size)

    def forward(self, state, action):
        x = None
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim= 1)
                a = action.flatten(start_dim=1)
                x = torch.cat((x.float(), a.float()), dim=1)
                x = self.mu_layers[i](x.float())
                x = torch.relu(x)
            else:
                raise ValueError("layer is not defined")
        mu_head = self.mu_head(x.float())

        x = None
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim= 1)
                a = action.flatten(start_dim=1)
                x = torch.cat((x.float(), a.float()), dim=1)
                x = self.var_layers[i](x.float())
                x = torch.relu(x)
            else:
                raise ValueError("layer is not defined")
        var_head = F.softplus(self.var_head(x.float())) + 1e-6
        return mu_head, var_head

class UncertaintyNN(nn.Module):
    def __init__(self, state_shape, action_shape, layers_type, layers_features):
        # state : B, state_size(linear)
        # action: A
        super(UncertaintyNN, self).__init__()
        self.layers_type = layers_type
        self.layers = []
        state_size = state_shape[1]
        action_size = action_shape

        for i, layer in enumerate(layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    linear_input_size = state_size + action_size
                    layer = nn.Linear(linear_input_size, layers_features[i])
                else:
                    layer = nn.Linear(layers_features[i - 1] + action_size, layers_features[i])
                self.add_module('hidden_layer_' + str(i), layer)
                self.layers.append(layer)
            else:
                raise ValueError("layer is not defined")
        if len(layers_type) > 0:
            self.head = nn.Linear(layers_features[-1], 1)
        else:
            self.head = nn.Linear(state_size + action_size, 1)

    def forward(self, state, action):
        x = None
        for i, layer in enumerate(self.layers_type):
            if layer == 'conv':
                raise NotImplemented("convolutional layer is not implemented")
            elif layer == 'fc':
                if i == 0:
                    x = state.flatten(start_dim= 1)
                a = action.flatten(start_dim=1)
                x = torch.cat((x.float(), a.float()), dim=1)
                x = self.layers[i](x.float())
                x = torch.tanh(x)
            else:
                raise ValueError("layer is not defined")
        if len(self.layers_type) > 0:
            head = torch.sigmoid(self.head(x.float()))
        else:
            x = state.flatten(start_dim= 1)
            a = action.flatten(start_dim=1)
            x = torch.cat((x.float(), a.float()), dim=1)
            head = torch.sigmoid(self.head(x.float()))
        return head