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


def produce_perturbations(model_params: list[torch.nn.parameter.Parameter]):
    """
        Produces 2 perturbations given the model_params. 

        :param model_params: list of parameters (in the form of tensors) of our model.


    """

    def _perturbation_vector():
        """
            Produces one perturbation vector per tensor in params.

            :returns: List of perturbation vectors where each entry is the perturbation vector
            for each tensor from the model params
        """
        return [np.random.rand(*(p.detach().numpy().shape)) for p in model_params]

    pos_perturb = _perturbation_vector()
    neg_perturb = [np.negative(p) for p in pos_perturb]
    return pos_perturb, neg_perturb



