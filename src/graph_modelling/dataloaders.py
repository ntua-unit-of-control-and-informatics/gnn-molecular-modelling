from rdkit import Chem
import sys
import random
import pandas as pd

if '../..' not in sys.path:
    sys.path.append('../..')

from utils.utils import class_balanced_random_split
from utilities import StandardNormalizer

import torch
from torch_geometric.data import Data
import torch.nn.functional as F


from torch.utils.data import Dataset
# from tqdm.notebook import tqdm

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split





def stratified_random_split_regression(df, num_bins, stratify_column, test_size=0.15, seed=None):
    
    discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='quantile')
    df['binned'] = discretizer.fit_transform(df[[stratify_column]])
    
    df_train, df_test = train_test_split(df, test_size=test_size, stratify=df['binned'], random_state=seed)
    
    df_train = df_train.drop(columns=["binned"])
    df_test = df_test.drop(columns=["binned"])

    
    return df_train, df_test

def endpoint_target_mean_std(endpoint_name):

    match endpoint_name:
        case 'hc20':
            return -0.123, 1.421
        case _:
            return 0.0, 1.0






def read_data(dataset_filepath, seed, test_split_percentage, endpoint_name, task, target_normalizer=None, shuffle=True):

    # X, y = [], []
    # sdf_supplier = Chem.SDMolSupplier(str(dataset_filepath))

    # for mol in sdf_supplier:
    #     if mol is not None:
    #         smiles = mol.GetProp('SMILES')
    #         # smiles = Chem.MolToSmiles(mol)
    #         ready_biodegradability = int(mol.GetProp('ReadyBiodegradability'))

    #         X.append(smiles)
    #         y.append(ready_biodegradability)
    
    df = pd.read_csv(dataset_filepath)
    
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    X = df[['SMILES']].values.ravel().tolist()
    y = df[[endpoint_name]].values.ravel().tolist()

    # if shuffle:
    #     zipped = list(zip(X, y))
    #     random.shuffle(zipped)
    #     X, y = zip(*zipped)
    #     X = list(X)
    #     y = list(y)

    if task == 'binary':
        X_train, X_test, y_train, y_test = class_balanced_random_split(X, y, seed=seed, test_ratio_per_class=test_split_percentage)
    elif task == 'regression':
        df_train, df_test = stratified_random_split_regression(df=df, num_bins=30, stratify_column=endpoint_name, test_size=test_split_percentage, seed=seed)
        X_train = df_train[['SMILES']].values.ravel().tolist()
        y_train = df_train[[endpoint_name]].values.ravel().tolist()
        X_test = df_test[['SMILES']].values.ravel().tolist()
        y_test = df_test[[endpoint_name]].values.ravel().tolist()
        if target_normalizer:
            y_train = target_normalizer(torch.tensor(y_train)).tolist()
            y_test = target_normalizer(torch.tensor(y_test)).tolist()
    else:
        raise ValueError(f"Unsupported task type '{task}'")


    
    


    Symb , Hs , Impv , Fc , Hb , ExpV , Deg = set() , set() , set() , set() , set() , set() , set()
    bond_type , conj = set() , set()

    # TODO: only from train
    for smile in X:
        for atom in Chem.MolFromSmiles(smile).GetAtoms():
            Symb.add(atom.GetSymbol())
            Hs.add(atom.GetTotalNumHs())
            Impv.add(atom.GetImplicitValence())
            Fc.add(atom.GetFormalCharge())
            Hb.add(atom.GetHybridization())
            ExpV.add(atom.GetExplicitValence())
            Deg.add(atom.GetDegree())
            
        for bond in Chem.MolFromSmiles(smile).GetBonds():
            bond_type.add(bond.GetBondType())
            conj.add(bond.GetIsConjugated())
    

    Symbols = list(Symb)
    Degree = list(Deg)
    Hs_Atoms = list(Hs)
    Implicit_Val = list(Impv)
    Formal_Charge = list(Fc)
    Explicit_Val = list(ExpV)
    Hybridization = list(Hb)

    structural_data = (Symbols, Degree, Hs_Atoms, Implicit_Val, Formal_Charge, Explicit_Val, Hybridization)
    

    train_dataset = GraphDataset(X_train, y_train, structural_data)
    test_dataset = GraphDataset(X_test, y_test, structural_data)

    return train_dataset, test_dataset


    




def atom_feature(atom, structural_data):
    Symbols, Degree, Hs_Atoms, Implicit_Val, Formal_Charge, Explicit_Val, Hybridization = structural_data
    return torch.tensor(one_of_k_encoding_unk(atom.GetSymbol(), Symbols) +
                        one_of_k_encoding(atom.GetTotalNumHs(), Hs_Atoms) +
                        one_of_k_encoding(atom.GetDegree(), Degree) +
                        one_of_k_encoding(atom.GetImplicitValence(), Implicit_Val) +
                        one_of_k_encoding(atom.GetFormalCharge(), Formal_Charge) +
                        one_of_k_encoding(atom.GetHybridization(), Hybridization) +
                        one_of_k_encoding(atom.GetExplicitValence(), Explicit_Val) +
                        [atom.GetIsAromatic()], dtype=torch.int16)

    

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f'input{x} not in allowable set{allowable_set}')
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x==s, allowable_set))


def nodes_and_adjacency(smile, y, structural_data):

    mol = Chem.MolFromSmiles(smile)
    # Create node Features
    feats = []
    for atom in mol.GetAtoms():
        feats.append(atom_feature(atom, structural_data)) # Get the 5 feats in a single atom of a mol
    mol_node_features = torch.stack(feats).float() # Stack them in an array [num_nodes x atom_features]

    # Create Adjacency Matrix
    ix1, ix2 = [], []

    for bond in mol.GetBonds():

        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ix1 += [start, end]
        ix2 += [end, start]

    adj_norm = torch.asarray([ix1, ix2], dtype=torch.int64) # Needs to be in COO Format

    return Data(x=mol_node_features,
                edge_index=adj_norm,
                y=y)
    




class GraphDataset(Dataset):

    def __init__(self, smiles, y, structural_data, transform=None):
        super(GraphDataset, self).__init__()
        
        self.transform = transform
        dataset_info = [nodes_and_adjacency(smile, y, structural_data) for smile, y in (zip(smiles, y))]

        self.df = [info for info in dataset_info]
        
        
    def __getitem__(self, idx):
        
        data = self.df[idx]

        
        if self.transform is not None:
            raise NotImplementedError("Haven't implemented or tested transforms")
            data = self.transform(data)
            
        return data

    def __len__(self):
        return len(self.df)
    

