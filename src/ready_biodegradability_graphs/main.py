from arguments import get_args_parser
from dataloaders import read_data
from train import train
from test import test
from utilities import initialize_graph_model, initialize_optimizer, check_gpu_availability

from pathlib import Path
import sys
import json


import torch
import random
import numpy as np
import torch.nn as nn

import warnings
# import copy

from torch.utils.data import Subset
from sklearn.model_selection import KFold


import logging
from typing import Iterable
import os
import datetime
import csv


# from torch.optim import lr_scheduler
# from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader

if '../..' not in sys.path:
    sys.path.append('../..')


# from models.graph_convolutional_network import GraphConvolutionalNetwork
# from models.graph_attention_network import GraphAttentionNetwork
# from utils.utils import class_balanced_random_split
from utils.loss import LabelSmoothingBCEWithLogitsLoss




# current_dir = Path().absolute()
warnings.filterwarnings("ignore")

# if str(current_dir.parent/'src') not in sys.path:
#     sys.path.append(str(current_dir.parent/'src'))


if __name__ == '__main__':


    # Parse arguments
    parser = get_args_parser()
    args = parser.parse_args()

    


    if isinstance(args.attention_heads, Iterable):
        if len(args.attention_heads) == 1:
            args.attention_heads, = args.attention_heads
    
    if isinstance(args.dropout, Iterable):
        if len(args.dropout) == 1:
            args.dropout, = args.dropout


    if args.inference:
        args.load_model_filepath = Path(args.load_model_filepath).resolve()


    if args.inference:
        raise NotImplementedError("Inference mode not implemented yet.")

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)


    # Dirs
    endpoint_name = 'ready_biodegradability'
    working_dir = Path.cwd()
    filename = 'AllPublicnew.sdf'
    data_dir = working_dir.parent.parent/'data'/endpoint_name
    dataset_filepath = data_dir/filename


    model_dir_prefix = "Model_"
    experiments_dir = working_dir.parent.parent/'experiments'/'graph_models'/endpoint_name


    if not args.inference: 
        # Create directories for train logs
        if os.path.exists(experiments_dir):
            model_dirs = [model_dir if os.path.isdir(os.path.join(experiments_dir, model_dir)) else None for model_dir in os.listdir(experiments_dir)]
            model_dirs = list(filter(None, model_dirs))
            ids = [int(dd.replace(model_dir_prefix,"")) if (model_dir_prefix) in dd and dd.replace(model_dir_prefix,"").isnumeric() else None for dd in model_dirs]
            ids = list(filter(None, ids))
            new_id = str(max(ids) + 1) if ids else "1"
        else:
            experiments_dir.mkdir(parents=True)
            new_id = "1"

        new_model_dir = experiments_dir.joinpath(model_dir_prefix + new_id)
        new_model_dir.mkdir()


    # Logging
    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
    
    if not args.inference: # both stdout and log file for training
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(new_model_dir, f'trainlogs_{new_id}.log'))]
    else:  # only stdout for inference
        handlers=[logging.StreamHandler(sys.stdout)]

    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)


    
    if args.inference:
        logging.info('Inference Mode\n')
    else:
        logging.info(f'{model_dir_prefix}ID: {new_id}\n')
        current_datetime = datetime.datetime.now()
        logging.info("Date: %s", current_datetime.strftime("%Y-%m-%d"))
        logging.info("Time: %s", current_datetime.strftime("%H:%M:%S"))


    if not args.inference:
        args_dict = vars(args)
        # Write dictionary to a JSON file
        with open(new_model_dir/'args.json', 'w') as json_file:
            json.dump(args_dict, json_file)
            

    # Load data
    train_val_dataset, test_dataset = read_data(dataset_filepath, args.seed, args.test_split_percentage)
    input_dim = train_val_dataset.df[0].x.shape[1]

    


    # Device
    device = torch.device('cuda:0' if not check_gpu_availability(not args.no_cuda) else 'cpu')

    torch.manual_seed(args.seed)
    if check_gpu_availability(not args.no_cuda):
        logging.info(f"\nDevice: \n- {torch.cuda.get_device_name()}")
        torch.cuda.manual_seed(args.seed)
    else:
        logging.info(f"\nDevice: {'CPU'}")

    
    
    # Logging Arguments
    logging.info("\nArguments:")
    logging.info('\n'.join(f'- {k}: {v}' for k, v in vars(args).items()))
    # logging.info('\n'.join(f'args.{k}={v}' for k, v in vars(args).items()))
    logging.info('\n')


    # Loss
    pos_weight = torch.tensor(args.loss_weights[1]/args.loss_weights[0]).to(device) if args.loss_weights[0] != args.loss_weights[1] else None
    loss_fn = LabelSmoothingBCEWithLogitsLoss(smoothing=args.smoothing, pos_weight=pos_weight)

    # Dataloaders kwargs
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if check_gpu_availability(not args.no_cuda) else {}
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # network kwargs
    model_kwargs = {
                'input_dim': input_dim,
                'hidden_dims': args.hidden_dims,
                'heads': args.attention_heads,
                'output_dim': 1,
                'activation': nn.ReLU(),
                'dropout': args.dropout,
                'graph_norm': args.graph_norm,
                'pooling': args.pooling                                   
            }
    
    optimizer_kwargs = {
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'betas': (args.beta1, args.beta2),
        'eps': args.adam_epsilon
    }


    if args.cross_validation: # cross-validation
        train_losses = np.zeros((args.cv_folds, args.n_epochs))
        val_losses = np.zeros((args.cv_folds, args.n_epochs))

        val_metrics_all = []
        

        kfold = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        for fold_idx, (train_index, val_index) in enumerate(kfold.split(train_val_dataset)):

            logging.info(f'CV Fold {fold_idx+1}:\n')


            model = initialize_graph_model(args.graph_network_type, model_kwargs).to(device)
            optimizer = initialize_optimizer(args.optimizer, model.parameters(), optimizer_kwargs)
            
        

            best_epoch = 1
            optimization_metric = 'f1'
            best_optimization_metric = float('inf') if optimization_metric == 'loss' else -float('inf')
            best_model_params = None

            val_metrics_all.append([])

            # Train-Val splits for this CV Fold
            train_dataset = Subset(train_val_dataset, train_index)
            val_dataset = Subset(train_val_dataset, val_index)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

            for epoch in range(1, args.n_epochs + 1):
                train_loss = train(epoch, args.n_epochs, train_loader, model, loss_fn, optimizer, device, use_tqdm=not args.no_tqdm)
                val_loss, val_metrics, val_conf_mat = test(val_loader, model, loss_fn, device)
                
                train_losses[fold_idx, epoch-1] = train_loss
                val_losses[fold_idx, epoch-1] = val_loss
                val_metrics_all[fold_idx].append(val_metrics)

                # val_metrics_all.append(val_metrics)
                logging.info(f"Epoch [{epoch}/{args.n_epochs}]:")
                
                epoch_logs = "  " + f"Train Loss: {train_loss:.4f}" + ' | '
                epoch_logs += f"Val Loss: {val_loss:.4f}"  + ' | '
                epoch_logs += f"Accuracy: {val_metrics['accuracy']:.4f}" + ' | '
                epoch_logs += f"BA: {val_metrics['balanced_accuracy']:.4f}" + ' | '
                epoch_logs += f"F1: {val_metrics['f1']:.4f}" + ' | '
                epoch_logs += f"MCC: {val_metrics['mcc']:.4f}" + ' | '
                epoch_logs += f"ROC_AUC: {val_metrics['roc_auc']:.4f}"
                logging.info(epoch_logs)
                
                # if optimization_metric == 'loss':
                #     if val_metrics['loss'] < best_optimization_metric:
                #         best_optimization_metric = val_metrics['loss']
                #         best_model_params = copy.deepcopy(model.state_dict())
                #         best_epoch = epoch
                # else:  
                #     if val_metrics[optimization_metric] > best_optimization_metric:
                #         best_optimization_metric = val_metrics[optimization_metric]
                #         best_model_params = copy.deepcopy(model.state_dict())
                #         best_epoch = epoch
                    
            np.save(new_model_dir/'train_losses.npy', train_losses)
            np.save(new_model_dir/'val_losses.npy', val_losses)
            with open(new_model_dir/'val_metrics.csv', 'w', newline='') as csv_file:
                fieldnames = ['cv_fold'] + list(val_metrics_all[0][0].keys())
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for cv_fold, fold_data in enumerate(val_metrics_all, start=0):
                    for data_dict in fold_data:
                        data_dict['cv_fold'] = cv_fold
                        writer.writerow(data_dict)

            # print(f"\n\nBest model corresponding to validation {optimization_metric} was found at epoch {best_epoch}, with a validation {optimization_metric} of {best_optimization_metric:.4f}\n")

    else:
        if args.val_split_percentage == 0: # only train-test

            model = initialize_graph_model(args.graph_network_type, model_kwargs).to(device)
            optimizer = initialize_optimizer(args.optimizer, model.parameters(), optimizer_kwargs)

            train_losses = np.zeros((1, args.n_epochs))
            train_loader = DataLoader(train_val_dataset, batch_size=args.batch_size, shuffle=True)

            for epoch in range(1, args.n_epochs + 1):
                train_loss = train(epoch, args.n_epochs, train_loader, model, loss_fn, optimizer, device, not args.no_tqdm)
                train_losses[1, epoch-1] = train_loss
            
            np.save(new_model_dir/'train_losses.npy', train_losses)        
            
        else: # train-val-test
            model = initialize_graph_model(args.graph_network_type, model_kwargs).to(device)
            optimizer = initialize_optimizer(args.optimizer, model.parameters(), optimizer_kwargs)

            train_dataset = ...
            val_dataset = ...
            
            raise NotImplementedError("Not implemented yet for val_split_percentage > 0")
        
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            
    

    

    # Refit model on Train and Val
    if args.refit and (args.cross_validation or ((not args.cross_validation) and (args.val_split_percentage > 0))): # cv-> refit -> test or train-val -> refit -> test
        logging.info("\nRefitting model on both Train and Validation data...\n")
        model = initialize_graph_model(args.graph_network_type, model_kwargs).to(device)
        optimizer = initialize_optimizer(args.optimizer, model.parameters(), optimizer_kwargs)

        refit_train_losses = np.zeros(args.n_epochs)
        train_loader = DataLoader(train_val_dataset, batch_size=args.batch_size, shuffle=True)


        for epoch in range(1, args.n_epochs + 1):
            refit_train_loss = train(epoch, args.n_epochs, train_loader, model, loss_fn, optimizer, device, not args.no_tqdm)
            refit_train_losses[epoch-1] = refit_train_loss
        
        np.save(new_model_dir/'refit_train_losses.npy', refit_train_losses)
        logging.info("\nPerformance of refitted model on Test Set:")
    else:
        if args.cross_validation: # cv --> test on last fold
            logging.info("\nPerformance of final CV fold model on Test Set:")
        else:
            if args.val_split_percentage == 0: # only train-test
                logging.info("\nPerformance on Test Set:")
            else: # train-val-test (no refit)
                logging.info("\nPerformance of final CV-fold model on Test Set:")

    test_loss, test_metrics, test_conf_mat = test(test_loader, model, loss_fn, device)
    for metric_name, metric in test_metrics.items():
        logging.info(f'- {metric_name}: {metric:.4f}')
    
    tn, fp, fn, tp = test_conf_mat
    conf_mat_dict = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    fieldnames = list(test_metrics.keys()) + list(conf_mat_dict.keys())
    with open(new_model_dir/'test_metrics.csv', 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{**test_metrics, **conf_mat_dict}])

        
        
    # if not args.cross_validation:
    #     model.load_state_dict(best_model_params)

    #     test_loss, test_metrics, test_conf_mat = test(test_loader, model, loss_fn, device)

    #     print(f"\nPerformance on Test Set on best epoch={best_epoch}:")
    #     for metric_name, metric in test_metrics.items():
    #         print(f'- {metric_name}: {metric:.4f}')

        

    
    

    if not args.inference:
        logging.info(f'{model_dir_prefix}ID: {new_id}\n')
        finish_datetime = datetime.datetime.now()
        logging.info("End Time: %s", finish_datetime.strftime("%H:%M:%S"))

        

    [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]