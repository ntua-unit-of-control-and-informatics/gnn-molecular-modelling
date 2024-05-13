import torch

import sys
import torch.nn as nn
from abc import ABC, abstractmethod
import logging

if '../..' not in sys.path:
    sys.path.append('../..')

from models.graph_convolutional_network import GraphConvolutionalNetwork
from models.graph_attention_network import GraphAttentionNetwork
from models.graph_sage_network import GraphSAGENetwork
from models.graph_transformer_network import GraphTransformerNetwork
from models.residual_block import Resnet

import warnings

from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, LambdaLR

from doa import Leverage

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
    - graph_network_type (str): The type of graph network to initialize. Supported values are 'convolutional', 'attention', and 'sage', 'transformer'.
    - model_kwargs (dict): Keyword arguments to be passed to the graph neural network model constructor.

    Returns:
    - torch.nn.Module: The initialized graph neural network model.

    Example:
    >>> model_kwargs = {'input_dim': 44, 'hidden_dim': [32, 64]}
    >>> model = initialize_graph_model('convolutional', model_kwargs)
    """
    # model = Resnet(**model_kwargs)
    # Initialise Model
    if graph_network_type=='convolutional':
        model = GraphConvolutionalNetwork(**model_kwargs)
    elif graph_network_type=='attention':
        model = GraphAttentionNetwork(**model_kwargs)
    elif graph_network_type=='sage':
        model = GraphSAGENetwork(**model_kwargs)
    elif graph_network_type=='transformer':
        model = GraphTransformerNetwork(**model_kwargs)
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
            optimizer = torch.optim.SGD(model_parameters, momentum=0.9, **optimizer_kwargs)
        case _:
            raise ValueError(f"Unsupported optimizer type '{optimization_algorithm}'")
        
    return optimizer

def initialize_scheduler(scheduler_type, optimizer, scheduler_kwargs):

    match scheduler_type:
        case 'multistep':
            milestones = scheduler_kwargs['milestones']
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)   
            # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)         
        case None:
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
        case _:
            raise ValueError(f"Unsupported scheduler type '{scheduler_type}'")
        
    return scheduler

def initialize_doa(doa_type):
    """
    Initialize an optimizer for the given optimization algorithm.

    Args:
    - doa_type (str): The Domain of Applicability algorithm to use. Supported values are 'Leverage'.

    Returns:
    - GraphEmbeddingSpaceDoA: The initialized DoA object.

    Example:
    >>> doa = initialize_optimizer('Leverage')
    """
    match doa_type:
        case 'Leverage':
            doa = Leverage()
        case _:
            raise ValueError(f"Unsupported doa type '{doa_type}'")
        
    return doa



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




def log_metrics(task, train_loss, metrics_output_tuple=None):

    if task == 'binary':
        val_loss, val_metrics, _ = metrics_output_tuple
        epoch_logs = "  " + f"Train Loss: {train_loss:.4f}" + ' | '
        epoch_logs += f"Val Loss: {val_loss:.4f}"  + ' | '
        epoch_logs += f"Accuracy: {val_metrics['accuracy']:.4f}" + ' | '
        epoch_logs += f"BA: {val_metrics['balanced_accuracy']:.4f}" + ' | '
        epoch_logs += f"F1: {val_metrics['f1']:.4f}" + ' | '
        epoch_logs += f"MCC: {val_metrics['mcc']:.4f}" + ' | '
        epoch_logs += f"ROC_AUC: {val_metrics['roc_auc']:.4f}"
        logging.info(epoch_logs)
    elif task == 'regression':
        val_loss, val_metrics = metrics_output_tuple
        epoch_logs = "  " + f"Train Loss: {train_loss:.4f}" + ' | '
        epoch_logs += f"Val Loss: {val_loss:.4f}"  + ' | '
        epoch_logs += f"Explained Variance: {val_metrics['explained_variance']:.4f}" + ' | '
        epoch_logs += f"R2: {val_metrics['r2']:.4f}" + ' | '
        epoch_logs += f"MSE: {val_metrics['mse']:.4f}" + ' | '
        epoch_logs += f"RMSE: {val_metrics['rmse']:.4f}" + ' | '
        epoch_logs += f"MAE: {val_metrics['mae']:.4f}" + ' | '
        logging.info(epoch_logs)
    else:
        raise ValueError(f"Unsupported task type '{task}'")



class LabelSmoothingBCEWithLogitsLoss(nn.Module):
    def __init__(self,
                 smoothing=[0.0, 0.0],
                 weight=None,
                 size_average=None,
                 reduce=None,
                 reduction='mean',
                 pos_weight=None,):
        
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(weight, size_average, reduce, reduction, pos_weight)
        

    def forward(self, outputs, targets):
        
        label_smoothing = [torch.tensor(self.smoothing[0]).float(), torch.tensor(1-self.smoothing[1]).float()]
        smoothed_labels = torch.where(targets < 0.5, label_smoothing[0], label_smoothing[1])
        loss = self.bce(outputs, smoothed_labels)
        
        return loss

def determine_mode(args):
        """
        Determines the mode of operation based on the provided arguments.

        Args:
            args: An object containing the command-line arguments or configuration parameters. 
                  It must have the following attributes:
                  - inference: A boolean indicating whether the mode is for inference.
                  - cross_validation: A boolean indicating whether cross-validation mode is enabled.
                  - cv_folds: An integer specifying the number of folds for cross-validation.
                  - val_split_percentage: A float specifying the percentage for train-validation split.
                  - refit: A boolean indicating whether the model should be refitted on the entire train-validation set before final testing.
        Returns:
            str: The mode of operation among: 'INFERENCE', 'CROSS_VALIDATION', CROSS_VALIDATION_REFIT', 'TRAIN_VAL_TEST', 'TRAIN_VAL_REFIT_TEST', 'TRAIN_TEST'.
        """
        if args.inference:
            return 'INFERENCE'
        
        if args.cross_validation:
            if args.cv_folds <= 1:
                raise ValueError("Number of cross-validation folds must be greater than 1.")
            return 'CROSS_VALIDATION_REFIT' if args.refit else 'CROSS_VALIDATION'

        if args.val_split_percentage > 0:
            return 'TRAIN_VAL_REFIT_TEST' if args.refit else 'TRAIN_VAL_TEST'
        
        if args.val_split_percentage == 0:
            return 'TRAIN_TEST'
        
        raise ValueError("Invalid arguments provided. Check your inputs.")

def mode2id(mode):
    """
    Converts a mode string to its corresponding ID.

    Args:
        mode (str): The mode string to convert.
    Returns:
        int: The ID corresponding to the given mode string.
             - 0 for 'INFERENCE'
             - 1 for 'CROSS_VALIDATION'
             - 2 for 'CROSS_VALIDATION_REFIT'
             - 3 for 'TRAIN_VAL_TEST'
             - 4 for 'TRAIN_VAL_REFIT_TEST'
             - 5 for 'TRAIN_TEST'
    """
    match mode:
        case 'INFERENCE':
            return 0
        case 'CROSS_VALIDATION':
            return 1
        case 'CROSS_VALIDATION_REFIT':
            return 2
        case 'TRAIN_VAL_TEST':
            return 3
        case 'TRAIN_VAL_REFIT_TEST':
            return 4
        case 'TRAIN_TEST':
            return 5
        case _:
            raise ValueError(f"Invalid mode {mode}")