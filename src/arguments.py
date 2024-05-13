import argparse
from typing import Iterable

def get_args_parser():

    parser = argparse.ArgumentParser(description='Molecular Graph Modelling with SMILES')

    parser.add_argument('--endpoint_name', metavar='ENDPOINT', type=str, required=True, help="Specifies the name of the endpoint for prediction.")
    parser.add_argument('--task', metavar='TASK', type=str, default='binary', choices=['binary', 'regression'])
    parser.add_argument('--data_dir', metavar='PATH', type=str, default=None, help='Specifies the path to the directory where the {endpoint_name}_dataset.csv file is located.')
    parser.add_argument('--batch_size', metavar='N', type=int, default=1, help="Specifies the number of samples in each mini-batch. Default is 1.")
    parser.add_argument('--n_epochs', metavar='N', type=int, default=100, help="Specifies the number of training epochs. Default is 100.")
    parser.add_argument('--optimizer', metavar='OPT', type=str, default='Adam', help="Optimization algorithm to use. Choices are 'Adam', 'AdamW', or 'SGD'. Default is 'Adam.", choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('--lr', metavar='LR', type=float, default=5e-4, help="Sets the learning rate for optimization. Requires 'Adam' or 'AdamW' optimizer. Default is 5e-4.")
    parser.add_argument('--weight_decay', metavar='DECAY', type=float, default=0.0, help="Sets the weight decay for regularization (L2 penalty). Default is 0.")
    parser.add_argument('--beta1', metavar='BETA1', type=float, default=0.9, help="Sets the exponential decay rate for the first moment estimates. Requires 'Adam' or 'AdamW' optimizers. Default is 0.9.")
    parser.add_argument('--beta2', metavar='BETA2', type=float, default=0.999, help=" Sets the exponential decay rate for the first moment estimates. Requires 'Adam' or 'AdamW' optimizer. Default is 0.999.")
    parser.add_argument('--adam_epsilon', metavar='EPSILON', type=float, default=1e-8, help="Sets the term added to the denominator to improve numerical stability in the optimization. Requires 'Adam' or 'AdamW' optimizers. Default is 1e-8.")
    parser.add_argument('--seed', metavar='S', type=int, default=1,  help="Sets the random seed for reproducibility. Default is 1.")
    # parser.add_argument('--load_model_id', type=int, default=None)
    parser.add_argument('--num_workers', metavar='N', type=int, default=0, help="Sets the number of worker processes for data loading. Default is 0.")
    # parser.add_argument('--evaluation_metric', type=str, default='f1_score')
    parser.add_argument('--graph_network_type', metavar='GTYPE', type=str, default='attention', choices=['convolutional', 'attention', 'sage', 'transformer'], help="Specifies the type of graph network to use. Choices are 'convolutional', 'attention', or 'sage'. Default is 'attention.")
    # parser.add_argument('--input_dim', type=int, default=44, help="")
    parser.add_argument('--hidden_dims', metavar='DIM', nargs="+", type=int, default=[32], help="Specifies the dimensions of hidden layers in the neural network. Default is [32].")
    parser.add_argument('--attention_heads', metavar='N', nargs="+", type=int, default=1, help="Specifies the number of attention heads in the attention mechanism for each layer. Default is 1.")
    parser.add_argument('--dropout', metavar='P', nargs="+", type=float, default=0.2, help="Specifies the dropout probabilities for each layer in the neural network. If a single value is provided, the same dropout will be applied across all layers. Default is 0.2.")
    parser.add_argument('--pooling', metavar='PL', type=str, default='mean', choices=['mean', 'add', 'max'], help="Specifies the type of pooling to be applied in the graph network. Choices are 'mean', 'add', or 'max'. Default is 'mean'.")
    parser.add_argument('--loss_weights', metavar='[W_NEGATIVE, W_POSITIVE]', nargs=2, type=float, default=[1.0, 1.0], help="Specifies the weights for the negative and possitive classes in the binary cross-entropy loss function. This argument is applicable only when the task is set to 'binary'. Default is [1.0, 1.0].")
    parser.add_argument('--smoothing', metavar='[S_NEGATIVE, S_POSITIVE]', nargs=2, type=float, default=[0.0, 0.0], help="Specifies the smoothing factors for the negative and possitive classes. The label for the negative and possitive classes will be smoothed towards S_NEGATIVE and 1 - S_POSSITIVE respectively. This argument is applicable only when the task is set to 'binary'. Default is [0.0, 0.0].")
    parser.add_argument('--cv_folds', metavar='N', type=int, default=5, help="Specifies the number of folds for cross-validation. Default is 5.")
    parser.add_argument('--val_split_percentage', metavar='P', type=float, default=0.15, help="Specifies the percentage of the train dataset to be used for validation. This argument is applicable only when cross_validation is set to False. Default is 0.15.")
    parser.add_argument('--test_split_percentage', metavar='P', type=float, default=0.15, help="Specifies the percentage of the dataset to be used for testing. Default is 0.15.")
    # parser.add_argument('--load_model_filepath', type=str, default='', help='Directory from which to load model')
    # parser.add_argument('--verbose', type=int, default=0)
    # parser.add_argument('--target_mean', type=float, default=0.0, help='the normalization mean for the target regression variable')
    # parser.add_argument('--target_std', type=float, default=1.0, help='the normalization mean for the target regression variable')
    parser.add_argument('--doa', metavar='DOA', type=str, default='Leverage', help="Specifies the Domain of Applicability algorithm")
    parser.add_argument('--inference', action='store_true', default=False, help="Flag to enable inference mode. Default is False.")
    parser.add_argument('--no_cuda', action='store_true', default=False, help="Flag to disable the use of CUDA for GPU acceleration. If set, the model will run on CPU only. Default is False.")
    parser.add_argument('--graph_norm', action='store_true', default=False, help="Flag to use graph normalization layers. Default is False.")
    parser.add_argument('--no_tqdm', action='store_true', default=False, help="Flag to disable the use of tqdm for progress bars. Default is False.")
    parser.add_argument('--refit', action='store_true', default=False, help="Flag to refit the model on the entire training dataset, including validation dataset, before testing. Default is False.")
    parser.add_argument('--cross_validation', action='store_true', default=False, help="Flag to enable cross-validation. Default is False.")
    parser.add_argument('--normalize_target', action='store_true', default=False, help="Flag to enable normalization of the target variable. This flag is applicable only when the task is set to 'regression'. Default is False.")


    return parser

def validate_arguments(args):

    if isinstance(args.attention_heads, Iterable):
        if len(args.attention_heads) == 1:
            args.attention_heads, = args.attention_heads
    
    if isinstance(args.dropout, Iterable):
        if len(args.dropout) == 1:
            args.dropout, = args.dropout
    
    return args

