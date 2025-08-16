"""
Implementation of population methods
"""

from utils import run_episode, test_policy, AdaptativeStdReduction
import torch
import logging
from policy import ParametricPolicy
from tqdm import tqdm
from functools import partial
import torch.multiprocessing as mp
import numpy as np


def _produce_perturbations(model_params: list[torch.nn.parameter.Parameter], n=10, std: float = 1.0):
    """
        Produces n perturbations given the model_params. 

        :param model_params: list of parameters (in the form of tensors) of our model.
        :returns: positive_perturbation, negative_perturbation
    """

    def _perturbation():
        """
            Produces one perturbation vector per tensor in params.

            :returns: List of perturbation vectors in the form of tensors where 
            each entry is the perturbation vector for each tensor from the model params
        """
        def p_generator(param): return torch.normal(
            mean=0.0,
            std=std,
            size=param.shape,
            device=param.device,
            dtype=param.dtype
        )
        return [p_generator(param) for param in model_params]

    return [_perturbation() for i in range(n)]


def _worker_function(args):
    # NOTE:
    # this approach creates copy of the policy for each perturbations, little inefficient as we should
    # create a copy for each worker (each worker evaluating multiple policies). But the code for this
    # less readable, so we took a simpler approach. Creating copies is also negligible compared to the
    # evaluation of the policy.
    env, policy, perturbation, runs_per_episode = args
    return test_policy(env=env, policy=policy.copy(), p_weights=perturbation, runs_per_episode=runs_per_episode)


def parallel_reward_computation(env, policy, perturbations, runs_per_episode, num_workers=None):
    """
    Parallelize reward computation using torch.multiprocessing
    Runs policy evaluation in //. 
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    rewards = np.zeroes
    args_list = [(env, policy, pert, runs_per_episode)
                 for pert in perturbations]

    with mp.Pool(num_workers) as pool:
        rewards = pool.map(
            _worker_function,
            args_list
        )
    return rewards


def train_population(env,
                     nb_episodes,
                     number_evaluation=1,
                     n=10,
                     adaptive_std=AdaptativeStdReduction(),
                     log_file_name="population.txt"):
    """
        Trains a policy with population method

        :param env: Is the gym environment
        :param nb_episodes: is the number of episodes on which or model is trained
        :param number_evaluation: is the number of times a given perturbed policy is evaluated. 
            Final reward of that policy is the average over the number of runs. 
        :param n: is the number of produced perturbations
        :param std: is the standard deviation of the perturbation
        :param log_file_name: name of the log file the output has to be saved to

    """
    logging.basicConfig(
        filename=f"logs/{log_file_name}", level=logging.INFO, format='%(message)s')
    policy: ParametricPolicy = ParametricPolicy(
        hidden_dims=64, requires_grad=False)

    std, _ = adaptive_std.get_std(reward=None)

    with torch.no_grad():
        for i in tqdm(range(nb_episodes), desc="Training Episodes", unit="episode"):
            # need some small std. Std = 1 produces huge differences
            perturbations = _produce_perturbations(
                policy.get_parameters(), std=std, n=n)
            rewards = [test_policy(env, policy, perturbation, number_evaluation)
                       for perturbation in perturbations]

            # NOTE: // computation is too slow with small values -> High overhead
            # rewards = parallel_reward_computation(env, policy, perturbations, runs_per_episode, num_workers=8)

            # select the best perturbation, which is the one leading to the best reward
            best_p = perturbations[rewards.index(max(rewards))]

            # add the best perturbation to the current weights
            updated_params = list(map(lambda t1, t2: t1 + t2,
                                      policy.get_parameters(),
                                      best_p))
            policy.update_weights(updated_params)

            reward = run_episode(env, lambda obs: policy(obs))

            std, avg = adaptive_std.get_std(reward)

            logging.info(f"{i + 1} {reward}")

            tqdm.write( f"Episode {i + 1}/{nb_episodes}: Reward = {reward}, Avg = {avg}, Std = {std}")

    return policy
