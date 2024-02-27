from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random


def class_balanced_random_split(X, y, seed=None, test_ratio_per_class=0.15):
    """
    Class-balanced dataset split into train and test partitions.

    Args:
        X (array-like): array-like of data input data points
        y (array-like): array-like of labels
        seed (int, optional): Random seed (Default: None)
        test_ratio_per_class (float, optional): Percentage of test samples per class (Default: 0.15)

    Returns:
        (tuple):
            * X_train (array-like): array-list of train data points
            * X_test (array-like): array-list of test data points
            * y_train (array-like): array-like list of train labels
            * y_test (array-like): array-like of test labels
    """


    if isinstance(y, list):
        idx2label = y
    elif isinstance(y, pd.DataFrame):
        idx2label = y.iloc[:, 0].tolist()
    elif isinstance(y, np.ndarray):
        idx2label = y
    else:
        raise TypeError(f"Unsupported type for y: {type(y)}")


    class_indices = {}
    for idx, label in enumerate(idx2label):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)


    train_indices = []
    test_indices = []
    for label, indices in class_indices.items():
        if len(indices) > 1:
            train_idx, test_idx = train_test_split(indices, test_size=test_ratio_per_class, random_state=seed)
        else:
            train_idx, test_idx = indices.copy(), []
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)
    
    random.shuffle(train_indices)
    random.shuffle(test_indices)


    if isinstance(X, list):
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
    elif isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        X_train.reset_index(inplace=True, drop=True)
        X_test.reset_index(inplace=True, drop=True)
    elif isinstance(X, np.ndarray):
        X_train = X[train_indices]
        X_test = X[test_indices]
    else:
        raise TypeError(f"Unsupported type for X: {type(X)}")


    if isinstance(y, list):
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]
    elif isinstance(y, pd.DataFrame):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
        y_train.reset_index(inplace=True, drop=True)
        y_test.reset_index(inplace=True, drop=True)
    elif isinstance(y, np.ndarray):
        y_train = y[train_indices]
        y_test = y[test_indices]
    else:
        raise TypeError(f"Unsupported type for y: {type(y)}")


    return X_train, X_test, y_train, y_test
