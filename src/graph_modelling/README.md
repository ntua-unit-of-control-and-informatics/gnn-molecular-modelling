# Molecular Graph Neural Network Modelling with SMILES

[Official Python implementation]

This repo section contains code for training Graph Neural Networks to predict various molecular properties, based on a molecule's SMILES representation.


## Overview

This Python script serves as a tool for simplifying the process of constructing and training graph neural networks. It automates model construction and training, allowing users to explore a wide range of hyperparameter options and values efficiently.

With this script, users can effortlessly vary hyperparameters across a large grid of options and values, facilitating model exploration and optimization.

## Usage
```bash
python main.py [-h] --endpoint_name ENDPOINT [--task TASK] [--data_dir DATA_DIR] [--batch_size N] [--n_epochs N]
               [--optimizer OPT] [--lr LR] [--weight_decay DECAY] [--beta1 BETA1] [--beta2 BETA2]
               [--adam_epsilon EPSILON] [--seed S] [--num_workers N] [--graph_network_type GTYPE]
               [--hidden_dims DIM [DIM ...]] [--attention_heads N [N ...]] [--dropout P [P ...]] [--pooling PL]
               [--loss_weights [W_NEGATIVE, W_POSITIVE] [W_NEGATIVE, W_POSITIVE]]
               [--smoothing [S_NEGATIVE, S_POSITIVE] [S_NEGATIVE, S_POSITIVE]] [--cv_folds N]
               [--val_split_percentage P] [--test_split_percentage P] [--inference] [--no_cuda] [--graph_norm]
               [--no_tqdm] [--refit] [--cross_validation] [--normalize_target]
```

### Train with Default Arguments
```bash
python main.py --endpoint_name ENDPOINT
```


### Inference with Default Arguments (not supported yet)
```bash
python main.py --endpoint_name ENDPOINT --inference --load_model_id [model_id]
```


### Arguments
`--endpoint_name ENDPOINT`: Specifies the name of the endpoint for prediction.<br>
`--task TASK`: Specifies the type of task. Choices are 'binary' for binary classification and 'regression' for regression. *Default is 'binary'.*<br>
`--data_dir PATH`: Specifies the path to the directory where the *{endpoint_name}_dataset.csv* file is located. *Default is None.*<br>
`--batch_size N`: Specifies the number of samples in each mini-batch. *Default is 1.*<br>
`--n_epochs N`: Specifies the number of training epochs. *Default is 100.*<br>
`--optimizer OPT`: Optimization algorithm to use. Choices are 'Adam', 'AdamW', or 'SGD'. *Default is 'Adam.*<br>
`--lr LR`: Sets the learning rate for optimization. Requires 'Adam' or 'AdamW' optimizer. *Default is 5e-4.*<br>
`--weight_decay DECAY`: Sets the weight decay for regularization (L2 penalty). *Default is 0.*<br>
`--beta1 BETA1`: Sets the exponential decay rate for the first moment estimates. Requires 'Adam' or  'AdamW' optimizers. *Default is 0.9.*<br>
`--beta2 BETA2`: Sets the exponential decay rate for the first moment estimates. Requires 'Adam' or 'AdamW' optimizer. *Default is 0.999.*<br>
`--adam_epsilon EPSILON`: Sets the term added to the denominator to improve numerical stability in the optimization. Requires 'Adam' or 'AdamW' optimizers. *Default is 1e-8.*<br>
`--seed N`: Sets the random seed for reproducibility. *Default is 1.*<br>
`--num_workers N`: Sets the number of worker processes for data loading. *Default is 0.*<br>
`--graph_network_type GTYPE` Specifies the type of graph network to use. Choices are 'convolutional', 'attention', or 'sage'. *Default is 'attention.*<br>
`--hidden_dims [DIM1,DIM2,...]`: Specifies the dimensions of hidden layers in the neural network. *Default is [32].*<br>
`--attention_heads [N1,N2,...] or N`: Specifies the number of attention heads in the attention mechanism for each layer. *Default is 1.*<br>
`--dropout [P1,P2,...] or P`: Specifies the dropout probabilities for each layer in the neural network. If a single value is provided, the same dropout will be applied across all layers. *Default is 0.2.*<br>
`--pooling PL`: Specifies the type of pooling to be applied in the graph network. Choices are 'mean', 'add', or 'max'. *Default is 'mean'.*<br>
`--loss_weights [W_NEGATIVE, W_POSITIVE]`: Specifies the weights for the negative and possitive classes in the binary cross-entropy loss function. This argument is applicable only when the task is set to 'binary'. *Default is [1.0, 1.0]*.<br>
`--smoothing [S_NEGATIVE, S_POSITIVE]`: Specifies the smoothing factors for the negative and possitive classes. The label for the negative and possitive classes will be smoothed towards S_NEGATIVE and 1 - S_POSSITIVE respectively. This argument is applicable only when the task is set to 'binary'. *Default is [0.0, 0.0].*<br>
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
`--normalize_target`: Flag to enable normalization of the target variable. This flag is applicable only when the task is set to 'regression'. *Default is False.*<br>



## Dataset Requirements

To train the molecular graph modeling project with SMILES, you need a dataset file in CSV format with the following specifications:

- The CSV file must contain at least two columns:
  - One column with the SMILES representation of molecules, labeled 'SMILES'.
  - One column with the target variable corresponding to the particular endpoint to be predicted. For example, if the endpoint name is 'hc20', the column name should be 'hc20'.
- The CSV file should be named following the convention: `{endpoint_name}_dataset.csv`. For instance, if the endpoint name is 'hc20', the dataset file should be named 'hc20_dataset.csv'.

Ensure that your dataset file meets these requirements before training the model.

### Dataset Location
If `data_dir` is provided by the user, the script will search for the CSV file in that directory. Otherwise, it will search at `../../data/{endpoint_name}` by default.


## Experiments Directory

During training, logs and metadata from the user's trained models are stored in an experiments directory. If the directory does not exist, it is created as `../../experiments/graph_models`.

Each set of experiments is specific to the corresponding endpoint, with a discrete experiments directory created for each endpoint. For example, the directory structure looks like this: `../../experiments/graph_models/{endpoint_name}`.


Within each endpoint-specific directory, there are individual directories for each model trained for that endpoint. These directories are named sequentially, such as `/Model_1`, `/Model_2`, and so on, with `{model_id}` indicating the unique identifier for each model.

Inside each model directory, the following logs and metadata are stored:
- `args.json`: Metadata containing the arguments used for training the model.
- `trainlogs_{model_id}.log`: Training logs for the specific model.
- `train_losses.npy`: Training loss values stored as a NumPy array.
- `val_losses.npy`: Validation loss values stored as a NumPy array.
- `val_metrics.csv`: Validation metrics saved in CSV format.
- `test_metrics.csv`: Test metrics saved in CSV format.

### Experiments Directory Structure
```
experiments/
└── graph_models/
    ├── {endpoint_name}/
    │   ├── Model_1/
    │   ├── Model_2/
    │   └── ...
    ├── {another_endpoint_name}/
    │   ├── Model_1/
    │   └── ...
    └── ...
```


This organized structure allows for easy management and retrieval of training logs and metadata for each model and endpoint.

Note: As a future feature, model saving functionality will be incorporated into this directory structure. Trained models will be saved within the corresponding model directories.


## Supported Evaluation Metrics

The model supports a range of evaluation metrics to assess its performance on both classification and regression tasks.

### Classification Metrics

For classification tasks, the following evaluation metrics are supported:

- **Accuracy:** Measures the proportion of correctly classified samples.
- **Balanced Accuracy:** Measures the accuracy of the model while accounting for class imbalances by averaging the accuracy of each class.
- **Precision:** Measures the proportion of true positive predictions among all positive predictions.
- **Recall:** Measures the proportion of true positive predictions among all actual positive samples.
- **F1 Score:** Harmonic mean of precision and recall, providing a balance between the two metrics.
- **ROC AUC Score:** Measures the area under the Receiver Operating Characteristic curve, indicating the model's ability to distinguish between classes.
- **Matthews Correlation Coefficient (MCC):** Measures the correlation between the predicted and actual binary classifications, considering both true and false positives and negatives.
- **Confusion Matrix:** A table representing the counts of true positive, false positive, true negative, and false negative predictions.


### Regression Metrics

For regression tasks, the following evaluation metrics are supported:

- **Mean Absolute Error (MAE):** Average of the absolute differences between predicted and actual values.
- **Mean Squared Error (MSE):** Average of the squared differences between predicted and actual values.
- **Root Mean Squared Error (RMSE):** Square root of the MSE, providing a measure of the average magnitude of errors.
- **R-squared (R2) Score:** Measures the proportion of variance in the target variable that is explained by the model.
- **Explained Variance Score:** Measures the proportion of variance in the target variable that is explained by the model, normalized by the variance of the target variable itself.




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



