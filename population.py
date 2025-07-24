"""
Implementation of population methods
"""
from utils import run_episode, test_policy
import torch
import logging
from policy import ParametricPolicy
from tqdm import tqdm

def _produce_perturbations(model_params: list[torch.nn.parameter.Parameter], n = 10, std: float = 1.0):
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
        p_generator = lambda param: torch.normal(
                mean=0.0, 
                std=std, 
                size=param.shape, 
                device=param.device, 
                dtype=param.dtype
        )
        return [p_generator(param) for param in model_params]

    return [_perturbation() for i in range(n)]

def train_population(env, 
                    nb_episodes, 
                    runs_per_episode = 1, 
                    n = 10, 
                    std=0.001, 
                    log_file_name="population.txt"):
    """
        Trains a policy with population method

        :param env: Is the gym environment
        :param nb_episodes: is the number of episodes on which or model is trained
        :param runs_per_episode: is the number of runs of every episode when evaluating 
            a certain perturbed policy.
        :param n: is the number of produced perturbations
        :param std: is the standard deviation of the perturbation
        :param log_file_name: name of the log file the output has to be saved to

    """
    logging.basicConfig(filename=f"logs/{log_file_name}", level=logging.INFO, format='%(message)s') 
    policy: ParametricPolicy = ParametricPolicy(requires_grad=False)

    with torch.no_grad():
        for i in tqdm(range(nb_episodes), desc="Training Episodes", unit="episode"):
            # need some small std. Std = 1 produces huge differences
            perturbations = _produce_perturbations(policy.get_parameters(), std=std)
            rewards = [test_policy(env, policy, perturbation, runs_per_episode) for perturbation in perturbations]

            # select the best perturbation, which is the one leading to the best reward
            best_p = perturbations[rewards.index(max(rewards))]

            # add the best perturbation to the current weights
            updated_params = list(map(lambda t1, t2: t1 + t2, 
                                      policy.get_parameters(), 
                                      best_p))
            
            policy.update_weights(updated_params)
            eval_reward = run_episode(env, lambda obs: policy(obs))
            logging.info(f"{i + 1} {eval_reward}")
            tqdm.write(f"Episode {i + 1}/{nb_episodes}: Reward = {eval_reward}")
    return policy

    pass
