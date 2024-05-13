# Automated Graph Neural Network Training for Molecular Property Prediction from SMILES


[Official Python implementation]

A repository with code concerning the automation of the modelling efforts for the PINK project.



**Project Leader**: 
Haralambos Sarimveis ([hsarimv@central.ntua.gr](mailto:hsarimv@central.ntua.gr))

**Contributors**: 
Giannis Pitoskas ([jpitoskas@gmail.com](mailto:jpitoskas@gmail.com)),
<!-- Giannis Savvas ([ioannis.savvas00@gmail.com](mailto:ioannis.savvas00@gmail.com)) -->


## Project Directory Structure

```
Project_dir/
    ├── data/
    ├── experiments/
    ├── models/
    └── src/
```

## Source Code
The `src/` directory contains the source code for training Graph Neural Networks (GNNs) using SMILES representations of molecules. For more detailed information about the source code and its usage, please refer to the internal README file located inside the `src/` directory.

## Models Implementation
The models/ directory contains class implementations for different types of graph neural networks (GNNs), designed to be easily configurable.

These implementations provide a flexible framework for constructing and training GNNs, allowing users to experiment with different architectures and hyperparameters to suit their specific needs.

## Data Directory
A `data/` directory is expected to be included in the project's root directory. This directory is intended to store datasets for different molecular properties (endpoints). 

Each property is organized into its own subdirectory, and dataset files follow a consistent naming convention:
- **Subdirectory Format**: Dataset subdirectories should follow the format `data/{property}/`
- **Naming Convention**: Dataset files should follow the format {property}_dataset.csv

An example is given below:

```
data/
├── propertyA/
│   └── propertyA_dataset.csv
├── propertyB/
│   └── propertyB_dataset.csv
│
└── ...
```


<!-- In the notebooks/ directory, the naming convention for notebooks follows the pattern `{category}_[...].ipynb`, where `{category}` serves as a prefix indicating the corresponding data directory. The `[...]` represents any arbitrary suffix. An example is given below:

 -->
<!-- 
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
└── ...
``` -->



