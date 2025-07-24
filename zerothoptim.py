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
from contextlib import contextmanager
import logging
from tqdm import tqdm

@contextmanager 
def temp_weights(policy: ParametricPolicy , new_params):
    """
        Applies temp weights to the policy. 
        Restores original weights once context is over
    """
    original_state = policy.state_dict()
    try:
        policy.update_weights(new_params)
        yield policy
    finally:
        policy.load_state_dict(original_state)

def run_episode(env, action_function):
    """
        produces one epsiode from our environment. selects an action from our 
        action function which should be a function which takes as argument and observation.

        :param env: our gym environment
        :param action_function: list of function using observations to produce a new action.
            providing multiple action function enables the training of policies for the same
            starting state.
    """
    # initial state
    observation, info = env.reset()
    episode_over = False
    total_reward = 0

    while not episode_over:
        # action = env.action_space.sample()
        action = action_function(observation)

        # take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

    return total_reward

def produce_perturbations(model_params: list[torch.nn.parameter.Parameter], std: float = 1.0):
    """
        Produces 2 perturbations given the model_params. 

        :param model_params: list of parameters (in the form of tensors) of our model.
        :returns: positive_perturbation, negative_perturbation
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

def train_0th_optim(env, 
                    nb_episodes, 
                    runs_per_episode = 1, 
                    lr = 0.001, 
                    std=0.001, 
                    log_file_name="0th-optim.txt"):
    """
        Trains a policy with 0-th order optimization

        :param env: Is the gym environment
        :param nb_episodes: is the number of episodes on which or model is trained
        :param runs_per_episode: is the number of runs of every episode when evaluating 
            a certain perturbed policy.
        :param log_file_name: name of the log file the output has to be saved to

    """
    logging.basicConfig(filename=f"logs/{log_file_name}", level=logging.INFO, format='%(message)s') 
    policy: ParametricPolicy = ParametricPolicy(requires_grad=False)

    def _test_policy(p_weights):
        """
            Takes original parameters of policy and the perturbation weights
            to produce a new policy. 
            Evaluates this policy.
            Restors original parameters
        """
        # new params are the original parameters changed by the perturbation weights.
        new_params = list(map(lambda t1, t2: t1 + t2, policy.get_parameters(), p_weights))
        with temp_weights(policy, new_params) as temp_policy:
            r = 0
            for i in range(runs_per_episode):
                r += run_episode(env, lambda obs: temp_policy(obs))
            r = r / runs_per_episode
        return r

    # reduces memory footprint as no backprop is used
    with torch.no_grad():
        for i in tqdm(range(nb_episodes), desc="Training Episodes", unit="episode"):
            # need some small std. Std = 1 produces huge differences
            perturbations = produce_perturbations(policy.get_parameters(), std=std)
            rewards = [_test_policy(perturbation) for perturbation in perturbations]

            # 0.5 * score of θ+ - score of θ-) × θ+
            gradient = list(map(lambda w: 0.5 * (rewards[0] - rewards[1]) * w, perturbations[0]))
            # apply learning rate to gradient and add this to the original parameters.
            updated_params = list(map(lambda t1, t2: t1 + t2, 
                                      policy.get_parameters(), 
                                      map(lambda w: w * lr, gradient)))
            
            policy.update_weights(updated_params)
            eval_reward = run_episode(env, lambda obs: policy(obs))
            logging.info(f"{i + 1} {eval_reward}")
            tqdm.write(f"Episode {i + 1}/{nb_episodes}: Reward = {eval_reward}")
    return policy
