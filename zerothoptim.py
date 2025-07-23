# TODO: Could look into perturbation: what if we use autocorrelation perturbation instead of random ?
"""
This file contains the code for Zeroth-order Optimization


We provide functions to create the perturbation vector which consists of random values
with average 0 and some variance. 
This perturbation vector is used to produce two perturbation of theta, our model parameters.
We then evaluate policy with the perturbated model.
"""

import torch
import numpy as np
from policy import ParametricPolicy

def train_epsiode(env, action_function):
    """
        Produces one epsiode from our environment. Selects an action from our 
        action function which should be a function which takes as argument and observation.

        :param env: our gym environment
        :param action_function: function using observations to produce a new action
    """
    # initial state
    observation, info = env.reset()
    episode_over = False
    total_reward = 0

    while not episode_over:
        # action = env.action_space.sample()
        action = action_function(observation)

        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

    return total_reward

def produce_perturbations(model_params: list[torch.nn.parameter.Parameter], std: float = 1.0):
    """
        Produces 2 perturbations given the model_params. 

        :param model_params: list of parameters (in the form of tensors) of our model.


    """

    def _perturbation_vector():
        """
            Produces one perturbation vector per tensor in params.

            :returns: List of perturbation vectors in the form of tensors where 
            each entry is the perturbation vector for each tensor from the model params
        """
        p_generator = lambda param: torch.normal(
                mean=0.0, 
                std=std, 
                size=param.shape, 
                device=param.device, 
                dtype=param.dtype
        )
        return [p_generator(param) for param in model_params]

    pos_perturb = _perturbation_vector()
    neg_perturb = [-p for p in pos_perturb]
    return pos_perturb, neg_perturb

def train_0th_optim(env, nb_episodes):
    policy: ParametricPolicy = ParametricPolicy(requires_grad=False)

    # reduces memory footprint as no backprop is used
    with torch.no_grad():
        for i in range(nb_episodes):
            p_p, n_p = produce_perturbations(policy.get_parameters())

            r = train_epsiode(env, lambda obs: policy(obs))
        print(r)
