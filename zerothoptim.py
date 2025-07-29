# TODO: Could look into perturbation: what if we use autocorrelation perturbation instead of random ?
# TODO: Could use simm anealing for std of perturbation. Could look at history of rewards, once hit 100 on average we reduce learning 
# Could check on replay buffer -> store all test run parameters 
"""
This file contains the code for Zeroth-order Optimization


We provide functions to create the perturbation vector which consists of random values
with average 0 and some variance. 
This perturbation vector is used to produce two perturbation of theta, our model parameters.
We then evaluate policy with the perturbated model.
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import logging
from tqdm import tqdm
from utils import run_episode, test_policy, AdaptativeStdReduction
from policy import ParametricPolicy

def _produce_perturbations(model_params: list[torch.nn.parameter.Parameter], std: float = 1.0):
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
                    number_evaluation = 1, 
                    lr = 0.001, 
                    adaptive_std = AdaptativeStdReduction(),
                    hidden_dims=128,
                    log_file_name="0th-optim.txt"
                    ):
    """
        Trains a policy with 0-th order optimization

        :param env: Is the gym environment
        :param nb_episodes: is the number of episodes on which or model is trained
        :param number_evaluation: is the number of times a given perturbed policy is evaluated. 
            Final reward of that policy is the average over the number of runs. 
        :param hidden_dims: is the number of units in the hidden dimension
        :param log_file_name: name of the log file the output has to be saved to
        :param adaptative_std: AdaptativeStdReduction object which changes std depending on rewards.

    """
    logging.basicConfig(filename=f"logs/{log_file_name}", level=logging.INFO, format='%(message)s') 
    policy: ParametricPolicy = ParametricPolicy(hidden_dims=hidden_dims, requires_grad=False)

    std, _ = adaptive_std.get_std(reward=None)

    # reduces memory footprint as no backprop is used
    with torch.no_grad():
        for i in tqdm(range(nb_episodes), desc="Training Episodes", unit="episode"):

            perturbations = _produce_perturbations(policy.get_parameters(), std=std)
            rewards = [test_policy(env, policy, perturbation, number_evaluation) for perturbation in perturbations]

            # 0.5 * score of θ+ - score of θ-) × θ+
            gradient = list(map(lambda w: 0.5 * (rewards[0] - rewards[1]) * w, perturbations[0]))
            # apply learning rate to gradient and add this to the original parameters.
            updated_params = list(map(lambda t1, t2: t1 + t2, 
                                      policy.get_parameters(), 
                                      map(lambda w: w * lr, gradient)))
            
            policy.update_weights(updated_params)

            reward = run_episode(env, lambda obs: policy(obs))
            std, avg = adaptive_std.get_std(reward)

            # logging.info(f"{i + 1} {reward}")
            # tqdm.write(f"Episode {i + 1}/{nb_episodes}: Reward = {reward}, Avg = {avg}, Std = {std}")
    reward_history = adaptive_std.rewards 
    average_size = 100
    logging.info(f"{np.average(reward_history[-average_size])}")
    #print(f"Average rewards over last {average_size} episodes {np.average(reward_history[-average_size])}")
    return policy
