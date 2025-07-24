import torch
import torch.nn as nn
import torch.nn.functional as F

class ParametricPolicy(nn.Module):
    """
        Parametric policy implementation for continuous control task. 
        Source - https://gymnasium.farama.org/environments/box2d/lunar_lander/

        :param states: 8 continuous input states for LunarLander-v3
            - Position `x`
            - Position `y`
            - Linear velocity `x`
            - Linear velocity `y`
            - Angle
            - Angular veclocity
            - Right leg ground
            - Left leg ground

        :param hidden_dims: width of the hidden layer

        :param action_dims: output dimensions which represents actions.
            1.  Throttle of main engine. Off if `main < 0`. Scales between 50% and 100% in `[0; 1]`
            2.  Throttle of lateral boosters. `[-1; 1]` with boosters off in region `-0.5 < booster < 0.5`. 
                Left booster scales 50%-100% between `[-1; -0.5]`.
                Right booster scales 50%-100% between `[0.5; 1]`.
        :param requires_grad: Boolean flag to control gradient computation (default: True)
    """
    def __init__(self, state_dims=8, hidden_dims=128, action_dims=2, requires_grad=True):
        super().__init__()

        self.input = nn.Linear(state_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, action_dims)

        # in case of Gradient-free optim gradient storing is not required (breaking news....)
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        """
            Forward pass for our Policy. 
            :param x: is an input state.
        """
        x = F.relu(self.input(x))
        # tanh restricts output between [-1; 1]
        action = torch.tanh(self.output(x))
        return action

    def update_weights(self, new_weights):
        """
            New weights should be a list of tensors which will be 
            assigned to the parameters of the network
        """
        for param, new_weights in zip(self.parameters(), new_weights):
            param.data = nn.parameter.Parameter(new_weights)

    def get_parameters(self):
        """
        Get policy parameters as a list of tensors.

        :returns: List of parameter tensors
        """
        return list(self.parameters())
