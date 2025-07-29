"""
Common functions used across multiple files
"""

from contextlib import contextmanager
from policy import ParametricPolicy
from dataclasses import dataclass
import numpy as np

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

def test_policy(env, policy, p_weights, number_evaluation):
    """
        Takes original parameters of policy and the perturbation weights
        to produce a new policy. Evaluates this policy. Restors original parameters
        :param policy: policy to use to run the epsiode
        :param p_weights: perturbation weights
        :param number_evaluation: is the number of times a given perturbed policy is evaluated. 
            Final reward of that policy is the average over the number of runs. 
    """
    # new params are the original parameters changed by the perturbation weights.
    new_params = list(map(lambda t1, t2: t1 + t2, policy.get_parameters(), p_weights))
    with temp_weights(policy, new_params) as temp_policy:
        r = 0
        for i in range(number_evaluation):
            r += run_episode(env, lambda obs: temp_policy(obs))
        r = r / number_evaluation
    return r

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

@dataclass
class AdaptativeStdReduction:
    """
        Class producing a new std value depending on a rolling average of past rewards.
        By default, the decay rate is set to 1, meaning that no std reduction is effectively applied.

        :param std: is the base standard deviation
        :param decay_rate: factor by which to multiple std each time the threshold is met.
        :param reward_treshold: reward target at which reduction starts.
        :param window_size: size of the rolling average.
    """
    std: float = 0.1
    decay_rate: float = 1.0
    reward_threshold: float = 100.0
    window_size: int = 5

    def __post_init__(self):
        self.std_max = self.std
        #  need some baseline std to avoid collapse of exploratin.
        self.std_min = self.std * 0.05
        self.rewards = np.array([])
    
    def get_std(self, reward):
        """
            Computes new std given current reward.
            :param reward: current reward to add to the rewards list 
                on which the rolling average is computed. If the average is 
                above the defined threshold defined during construction, std 
                reduction begins. If reward is `None` returns the stored std 
                without computing rolling average.
            :returns: (new_std, rolling window avg)
        """
        if reward is None:
            return self.std, None

        self.rewards = np.append(self.rewards, reward)

        i = len(self.rewards)
        x = 0 if i > self.window_size else self.window_size - i
        start = i - (self.window_size - x)

        avg = np.average(self.rewards[start: i]) 

        # only apply reduction with sufficent sample size
        if len(self.rewards) > self.window_size:
            if avg > self.reward_threshold:
                self.std = max(self.std * self.decay_rate, self.std_min)
            else: 
                self.std = min(self.std / self.decay_rate, self.std_max)
        return self.std, avg
