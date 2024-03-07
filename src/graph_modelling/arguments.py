import argparse
from typing import Iterable

def get_args_parser():

    parser = argparse.ArgumentParser(description='Ready Biodegradability Graph Modelling')

    
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size (default: 1)')
    parser.add_argument('--n_epochs', type=int, default=100, metavar='N',  help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimization algorithm (default: Adam)', choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay coefficient ')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 coefficient')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 coefficient')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--inference', action='store_true', default=False, help='inference mode (default: False)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # parser.add_argument('--load_model_id', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    # parser.add_argument('--evaluation_metric', type=str, default='f1_score')
    parser.add_argument('--graph_network_type', type=str, default='attention', choices=['convolutional', 'attention', 'sage'], help='type of graph network')
    parser.add_argument('--input_dim', type=int, default=44, help='feature vector dimension')
    parser.add_argument('--hidden_dims', nargs="+", type=int, default=[32], help='list of hidden dimensions')
    parser.add_argument('--attention_heads', nargs="+", type=int, default=1, help='list of number of attention heads, or a single integer if the same number of attention heads to be used across all layers')
    parser.add_argument('--dropout', nargs="+", type=float, default=0.2, help='list of dropout probabilities, or a single probability if the same dropout is to be applied across all layers')
    parser.add_argument('--graph_norm', action='store_true', default=False, help='graph normalization (default: False)')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'add', 'max'], help='type of pooling to be applied in the graph network')
    parser.add_argument('--loss_weights', nargs=2, type=float, default=[1.0, 1.0], help='loss weight per class')
    parser.add_argument('--smoothing', nargs=2, type=float, default=[0.0, 0.0], help='how much to smoothing to apply to each class label')
    parser.add_argument('--no_tqdm', action='store_true', default=False, help='disable/silence tqdm')
    parser.add_argument('--cross_validation', action='store_true', default=False, help='enables Cross-Validation')
    parser.add_argument('--cv_folds', type=int, default=5, help='number of cross-validation folds')
    parser.add_argument('--refit', action='store_true', default=False, help='refit on both train and validation data to evaluate on test set')
    parser.add_argument('--val_split_percentage', type=float, default=0.15, help='percentage of dataset to be used for validation')
    parser.add_argument('--test_split_percentage', type=float, default=0.15, help='percentage of dataset to be used for testing')
    # parser.add_argument('--load_model_filepath', type=str, default='', help='Directory from which to load model')
    # parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--endpoint_name', type=str, default='ready_biodegradability')
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'regression'])


    return parser

def validate_arguments(args):

    if isinstance(args.attention_heads, Iterable):
        if len(args.attention_heads) == 1:
            args.attention_heads, = args.attention_heads
    
    if isinstance(args.dropout, Iterable):
        if len(args.dropout) == 1:
            args.dropout, = args.dropout
    
    return args

