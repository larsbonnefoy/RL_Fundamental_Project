"""
Common functions used across multiple files
"""

from contextlib import contextmanager
from policy import ParametricPolicy

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

def test_policy(env, policy, p_weights, runs_per_episode):
    """
        Takes original parameters of policy and the perturbation weights
        to produce a new policy. Evaluates this policy. Restors original parameters
        :param policy: policy to use to run the epsiode
        :param p_weights: perturbation weights
        :param runs_per_episode: is the number of runs of every episode when evaluating 
            a certain perturbed policy.
    """
    # new params are the original parameters changed by the perturbation weights.
    new_params = list(map(lambda t1, t2: t1 + t2, policy.get_parameters(), p_weights))
    with temp_weights(policy, new_params) as temp_policy:
        r = 0
        for i in range(runs_per_episode):
            r += run_episode(env, lambda obs: temp_policy(obs))
        r = r / runs_per_episode
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
