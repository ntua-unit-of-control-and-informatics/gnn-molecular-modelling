import torch
import sys
import torch.nn as nn

if '../..' not in sys.path:
    sys.path.append('../..')

from models.graph_convolutional_network import GraphConvolutionalNetwork
from models.graph_attention_network import GraphAttentionNetwork

import warnings


def check_gpu_availability(use_gpu=True):
    """
    Check if CUDA is available and supported by PyTorch.

    Args:
    - use_gpu (bool): Whether to use GPU. Defaults to True.

    Returns:
    - bool: True if CUDA is available and supported, False otherwise.
    """

    cuda_available = torch.cuda.is_available()
    cuda_supported = torch.backends.cudnn.enabled

    if not cuda_supported:
        warnings.warn("CuDNN is disabled. This may affect the performance of certain operations.", RuntimeWarning)

    return use_gpu and cuda_available


def initialize_graph_model(graph_network_type, model_kwargs):
    """
    Initialize a graph neural network model based on the specified graph network type.

    Args:
    - graph_network_type (str): The type of graph network to initialize. Supported values are 'convolutional', 'attention', and 'sage'.
    - model_kwargs (dict): Keyword arguments to be passed to the graph neural network model constructor.

    Returns:
    - torch.nn.Module: The initialized graph neural network model.

    Example:
    >>> model_kwargs = {'input_dim': 44, 'hidden_dim': [32, 64]}
    >>> model = initialize_graph_model('convolutional', model_kwargs)
    """
    # Initialise Model
    if graph_network_type=='convolutional':
        model = GraphConvolutionalNetwork(**model_kwargs)
    elif graph_network_type=='attention':
        model = GraphAttentionNetwork(**model_kwargs)
    elif graph_network_type=='sage':
        raise NotImplementedError("Graph Sage Network not implemented yet.")
        model = ...
    else:
        raise ValueError(f"Unsupported graph network type for '{graph_network_type}'")

    return model


def initialize_optimizer(optimization_algorithm, model_parameters, optimizer_kwargs):
    """
    Initialize an optimizer for the given optimization algorithm.

    Args:
    - optimization_algorithm (str): The optimization algorithm to use. Supported values are 'Adam', 'AdamW', and 'SGD'.
    - model_parameters (iterable): The parameters of the model for which the optimizer will be initialized. It typically comes from model.parameters().
    - optimizer_kwargs (dict): Keyword arguments to be passed to the optimizer constructor.

    Returns:
    - torch.optim.Optimizer: The initialized optimizer.

    Example:
    >>> model = YourModel()
    >>> optimizer_kwargs = {'lr': 0.001, 'betas': (0.9, 0.999)}
    >>> optimizer = initialize_optimizer('Adam', model.parameters(), optimizer_kwargs)
    """
    match optimization_algorithm:
        case 'Adam':
            optimizer = torch.optim.Adam(model_parameters, **optimizer_kwargs)
        case 'AdamW':
            optimizer = torch.optim.AdamW(model_parameters, **optimizer_kwargs)
        case 'SGD':
            optimizer_kwargs.pop('betas', None)
            optimizer_kwargs.pop('eps', None)
            optimizer = torch.optim.SGD(model_parameters, **optimizer_kwargs)
        case _:
            raise ValueError(f"Unsupported optimizer type '{optimization_algorithm}'")
        
    return optimizer



class StandardNormalizer(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super(StandardNormalizer, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, input_tensor):
        """
        Args:
            input_tensor (Tensor): Input tensor of size (N,) to be normalized.
        Returns:
            Tensor: Normalized tensor.
        """
        normalized_tensor = input_tensor.sub_(self.mean).div_(self.std)
        return normalized_tensor
    
    def denormalize(self, normalized_tensor):
        """
        Args:
            normalized_tensor (Tensor): Normalized tensor of size (N,) to be denormalized.
        Returns:
            Tensor: Denormalized tensor.
        """
        denormalized_tensor = normalized_tensor.mul_(self.std).add_(self.mean)
        return denormalized_tensor

    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

