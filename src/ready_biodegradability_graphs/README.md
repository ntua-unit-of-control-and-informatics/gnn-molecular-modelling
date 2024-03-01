# Ready Biodegradability

[Official Python implementation]

Code for Ready Biodegradability prediction using Graph Neural Networks


### Train with Default Arguments
```bash
python main.py
```


### Inference with Default Arguments (not supported yet)
```bash
python /main.py --inference --load_model_id [model_id]
```


### Arguments
`--batch_size N`: Specifies the number of samples in each mini-batch. *Default is 1.*<br>
`--n_epochs N`: Specifies the number of training epochs. *Default is 100.*<br>
`--lr LR`: Sets the learning rate for optimization. Requires 'Adam' or 'AdamW' optimizer. *Default is 5e-4.*<br>
`--weight_decay DECAY`: Sets the weight decay for regularization. *Default is 0.*<br>
`--beta1 BETA1`: Sets the exponential decay rate for the first moment estimates. Requires 'Adam' or  'AdamW' optimizers. *Default is 0.9.*<br>
`--beta2 BETA2`: Sets the exponential decay rate for the first moment estimates. Requires 'Adam' or 'AdamW' optimizer. *Default is 0.999.*<br>
`--adam_epsilon EPSILON`: Sets the term added to the denominator to improve numerical stability in the optimization. Requires 'Adam' or 'AdamW' optimizers. *Default is 1e-8.*<br>
`--seed N`: Sets the random seed for reproducibility. *Default is 1.*<br>
`--num_workers N`: Sets the number of worker processes for data loading. *Default is 0.*<br>
`--graph_network_type attention` Specifies the type of graph network to use. Choices are 'convolutional', 'attention', or 'sage'. *Default is 'attention.*<br>
`--hidden_dims [DIM1,DIM2,...]`: Specifies the dimensions of hidden layers in the neural network. *Default is [32].*<br>
`--attention_heads [N1,N2,...] or N`: Specifies the number of attention heads in the attention mechanism for each layer. *Default is 1.*<br>
`--pooling PL`: Specifies the type of pooling to be applied in the graph network. Choices are 'mean', 'add', or 'max'. *Default is 'mean.*<br>
`--loss_weights [W_NEGATIVE, W_POSITIVE]`: Specifies the weights for the negative and possitive classes in the loss function. *Default is [1.0, 1.0]*.<br>
`--smoothing [S_NEGATIVE, S_POSITIVE]`: Specifies the smoothing factors for the negative and possitive classes. *Default is [0.0, 0.0].*<br>
`--cv_folds N`: Specifies the number of folds for cross-validation. *Default is 5.*<br>
`--val_split_percentage P`: Specifies the percentage of the train dataset to be used for validation. This argument is applicable only when cross_validation is set to False. *Default is 0.15.*<br>
`--test_split_percentage P`: Specifies the percentage of the dataset to be used for testing. *Default is 0.15.*<br>
<!-- `--load_model_filepath 0.2`<br>
`--verbose 0.2`<br> -->

### Flags
`--inference`: Flag to enable inference mode. *Default is False.*<br>
`--no_cuda`: Flag to disable the use of CUDA for GPU acceleration. If set, the model will run on CPU only. *Default is False.*<br>
`--graph_norm`: Flag to use graph normalization layers. *Default is False.*<br>
`--no_tqdm`: Flag to disable the use of tqdm for progress bars. *Default is False.*<br>
`--refit`: Flag to refit the model on the entire training dataset, including validation dataset, before testing. *Default is False.*<br>
`--cross_validation`: Flag to enable cross-validation. *Default is False.*<br>

<!-- 

### Project Directory Structure

```
Project_dir/
    ├── models/
    ├── notebooks/
    ├── utils/
    ├── src/
    └── data/
```

### Data Directory

In the notebooks/ directory, the naming convention for notebooks follows the pattern `{category}_[...].ipynb`, where `{category}` serves as a prefix indicating the corresponding data directory. The `[...]` represents any arbitrary suffix. An example is given below:


```
Project_dir/
│
├── notebooks/
│   ├── modellingA_[...].ipynb
│   ├── ...
│   └── other_notebook.ipynb
│
├── data/
│   ├── modellingA/
│   │   ├── data_file1.csv
│   │   ├── data_file2.csv
│   │   └── ...
│   ├── ...
│   └── other_data_dir/
│
└── ... -->
<!-- ``` -->



