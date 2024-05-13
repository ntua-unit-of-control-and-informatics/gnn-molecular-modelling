from rdkit import Chem
import sys
import pandas as pd
from pathlib import Path

# if '../..' not in sys.path:
#     sys.path.append('../..')
# if str(Path(__file__).resolve().parent.parent.parent) not in sys.path:
#     sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# from utils.utils import class_balanced_random_split

import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
# from tqdm.notebook import tqdm

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

import warnings
import json
import pickle

import numpy as np
import random


def class_balanced_random_split(X, y, test_ratio_per_class=0.15, seed=None):
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
            # warnings.warn(f"No mean and std available for target variable '{endpoint_name}'. Defaulting to mean=0.0, std=1.0")
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
        # import numpy as np
        # y_train = np.random.normal(-0.123, 1.421, len(y_train)).tolist()
        # y_test = np.random.normal(-0.123, 1.421, len(y_test)).tolist()
        # y_train = np.random.uniform(-1, 1, len(y_train)).tolist()
        # y_test = np.random.uniform(-1, 1, len(y_test)).tolist()
        
    else:
        raise ValueError(f"Unsupported task type '{task}'")


    inverse_hybridization_names = {v: k for k, v in Chem.rdchem.HybridizationType.names.items()}


    Symb , Hs , Impv , Fc , Hb , ExpV , Deg = set() , set() , set() , set() , set() , set() , set()
    Bonds = set() 

    TDeg = set()
    Chtag = set()

    # TODO: only from train
    for i, smile in enumerate(X_train):
        for atom in Chem.MolFromSmiles(smile).GetAtoms():
            Symb.add(atom.GetSymbol())
            Hs.add(atom.GetTotalNumHs())
            Impv.add(atom.GetImplicitValence())
            Fc.add(atom.GetFormalCharge())
            Hb.add(atom.GetHybridization())
            ExpV.add(atom.GetExplicitValence())
            Deg.add(atom.GetDegree())
            TDeg.add(atom.GetTotalDegree())
            Chtag.add(atom.GetChiralTag())
            
        for bond in Chem.MolFromSmiles(smile).GetBonds():
            Bonds.add(bond.GetBondType())
    

    Symbols = sorted(Symb)
    Degree = sorted(Deg)
    TDegree = sorted(TDeg)
    Hs_Atoms = sorted(Hs)
    Implicit_Val = sorted(Impv)
    Explicit_Val = sorted(ExpV)
    Formal_Charge = sorted(Fc)
    Hybridization = sorted(Hb)
    ChiralTag = sorted(Chtag)

    BondTypes = sorted(Bonds)

    # structural_data = (Symbols, Degree, Hs_Atoms, Implicit_Val, Formal_Charge, Explicit_Val, Hybridization)
    
    atom_allowable_set = {
        'symbol': Symbols,
        'degree': Degree,
        # 'total_degree': TDegree,
        'total_num_hs': Hs_Atoms,
        'implicit_valence': Implicit_Val,
        'formal_charge': Formal_Charge,
        'explicit_valence': Explicit_Val,
        'hybridization': Hybridization,
        'is_aromatic': None,
        # 'is_in_ring': None,
        # 'mass': None,
        # 'chiral_tag': ChiralTag
        # '_CIPCode': ['R','S']
    }

    bond_allowable_sets = {
        # "bond_type": BondTypes, # could be of type UNSPECIFIED
        "bond_type": [Chem.rdchem.BondType.SINGLE,
                      Chem.rdchem.BondType.DOUBLE,
                      Chem.rdchem.BondType.TRIPLE,
                      Chem.rdchem.BondType.AROMATIC], 
        "is_conjugated": None,
        "is_in_ring": None,
    }

    featurizer = SmilesGraphFeaturizer(include_edge_features=True, warnings_enabled=False)

    featurizer.set_atom_allowable_sets(atom_allowable_set)
    featurizer.set_bond_allowable_sets(bond_allowable_sets)

    # featurizer.save_config("featurizer_config.pkl")

    train_dataset = SmilesGraphDataset(X_train, y_train, featurizer=featurizer)
    train_dataset.precompute_featurization()

    # for i, f in enumerate(train_dataset.get_atom_feature_labels()):
    #     print(i, ":", f)

    test_dataset = SmilesGraphDataset(X_test, y_test, featurizer=featurizer)
    test_dataset.precompute_featurization()
        # test_dataset = SmilesGraphDataset(X_test, y_test).config_from_other_dataset(train_dataset)

    return train_dataset, test_dataset, featurizer


class SmilesGraphFeaturizer():
    
    @staticmethod
    def _get_CIPCODE(atom):
        try:
            cip = atom.GetProp('_CIPCode')
        except KeyError:
            cip = None
        return cip
    
    SUPPORTED_ATOM_CHARACTERISTICS = {"symbol": lambda atom: atom.GetSymbol(),
                                      "degree": lambda atom: atom.GetDegree(),
                                      "total_degree": lambda atom: atom.GetTotalDegree(),
                                      "formal_charge": lambda atom: atom.GetFormalCharge(),
                                      "num_radical_electrons": lambda atom: atom.GetNumRadicalElectrons(),
                                      "hybridization": lambda atom: atom.GetHybridization(),
                                      "is_aromatic": lambda atom: atom.GetIsAromatic(),
                                      "is_in_ring": lambda atom: atom.IsInRing(),
                                      "total_num_hs": lambda atom: atom.GetTotalNumHs(),
                                      "num_explicit_hs": lambda atom: atom.GetNumExplicitHs(),
                                      "num_implicit_hs": lambda atom: atom.GetNumImplicitHs(),
                                      "_CIPCode": lambda atom: SmilesGraphDataset._get_CIPCODE(atom),
                                      "_ChiralityPossible": lambda atom: atom.HasProp('_ChiralityPossible'),
                                      "isotope": lambda atom: atom.GetIsotope(),
                                      "total_valence": lambda atom: atom.GetTotalValence(),
                                      "explicit_valence": lambda atom: atom.GetExplicitValence(),
                                      "implicit_valence": lambda atom: atom.GetImplicitValence(),
                                      "chiral_tag": lambda atom: atom.GetChiralTag(), 
                                      "mass": lambda atom: (atom.GetMass()-14.5275)/9.4154}

    SUPPORTED_BOND_CHARACTERISTICS = {"bond_type": lambda bond: bond.GetBondType(),
                                      "is_conjugated": lambda bond: bond.GetIsConjugated(),
                                      "is_in_ring": lambda bond: bond.IsInRing(),
                                      "stereo": lambda bond: bond.GetStereo()}
    
    NON_ONE_HOT_ENCODED = [
        # "formal_charge",
                           "num_radical_electrons",
                           "is_aromatic",
                           "_ChiralityPossible",
                           "is_conjugated",
                           "is_in_ring",
                           "mass"]

    def __init__(self, include_edge_features=True, warnings_enabled=True):
        
        self.include_edge_features = include_edge_features
        
        self.warnings_enabled = warnings_enabled
        
        self.atom_allowable_sets = {}
        self.bond_allowable_sets = {}
        
    
    def get_supported_atom_characteristics(self):
        return self.SUPPORTED_ATOM_CHARACTERISTICS.keys()
    
    def get_supported_bond_characteristics(self):
        return self.SUPPORTED_BOND_CHARACTERISTICS.keys()
    
    def config_from_other_featurizer(self, featurizer):
        
        self.include_edge_features = featurizer.include_edge_features
        
        self.warnings_enabled = featurizer.warnings_enabled
        
        self.set_atom_allowable_sets(featurizer.atom_allowable_sets)

        self.set_bond_allowable_sets(featurizer.bond_allowable_sets)
        
        return self
        
    
    # def set_atom_characteristics(self, atom_characteristics):
    #     for key in atom_characteristics:
    #         if key not in self.SUPPORTED_ATOM_CHARACTERISTICS.keys(): 
    #             raise ValueError(f"Invalid atom characteristic '{key}'")
    #     self.atom_characteristics = list(atom_characteristics)
        
        
    # def set_bond_characteristics(self, bond_characteristics):
    #     for key in bond_characteristics:
    #         if key not in self.SUPPORTED_BOND_CHARACTERISTICS.keys(): 
    #             raise ValueError(f"Invalid bond characteristic '{key}'")
    #     self.bond_characteristics = list(bond_characteristics)
    
    
    def set_atom_allowable_sets(self, atom_allowable_sets_dict):
        
        self.atom_allowable_sets = dict()
        for atom_characteristic, allowable_set in atom_allowable_sets_dict.items():
            self.add_atom_characteristic(atom_characteristic, allowable_set)
    
    def set_bond_allowable_sets(self, bond_allowable_sets_dict):
        
        self.bond_allowable_sets = dict()
        for bond_characteristic, allowable_set in bond_allowable_sets_dict.items():
            self.add_bond_characteristic(bond_characteristic, allowable_set)

    def add_atom_characteristic(self, atom_characteristic, allowable_set=None):

        if atom_characteristic not in self.get_supported_atom_characteristics():
            raise ValueError(f"Unsupported atom characteristic '{atom_characteristic}'")
        
        if atom_characteristic in self.atom_allowable_sets.keys():
            self.warning(f"The atom allowable set for '{atom_characteristic}' will be overwritten.")

        if atom_characteristic in self.NON_ONE_HOT_ENCODED:
            if allowable_set is not None:
                self.warning(f"Atom allowable set given for '{atom_characteristic}' will be ignored (not one-hot encoded)")
            self.atom_allowable_sets[atom_characteristic] = None
        else:
            if allowable_set is None:
                self.warning(f"The atom allowable set for '{atom_characteristic}' is set to default.")
                allowable_set = self.get_default_atom_allowable_set(atom_characteristic)
            elif not isinstance(allowable_set, (list, tuple)):
                raise TypeError("Input dictionary must have values of type list, tuple or None.")
            self.atom_allowable_sets[atom_characteristic] = list(allowable_set)

    def add_bond_characteristic(self, bond_characteristic, allowable_set=None):

        if bond_characteristic not in self.get_supported_bond_characteristics():
            raise ValueError(f"Unsupported bond characteristic '{bond_characteristic}'")
        
        if bond_characteristic in self.bond_allowable_sets.keys():
            self.warning(f"The bond allowable set for '{bond_characteristic}' will be overwritten.")

        if bond_characteristic in self.NON_ONE_HOT_ENCODED:
            if allowable_set is not None:
                self.warning(f"Bond allowable set given for '{bond_characteristic}' will be ignored (not one-hot encoded)")
            self.bond_allowable_sets[bond_characteristic] = None
        else:
            if allowable_set is None:
                self.warning(f"The bond allowable set for '{bond_characteristic}' is set to default.")
                allowable_set = self.get_default_bond_allowable_set(bond_characteristic)
            if not isinstance(allowable_set, (list, tuple)):
                raise TypeError("Input dictionary must have values of type list, tuple or None.")
            self.bond_allowable_sets[bond_characteristic] = list(allowable_set)
    

    def get_atom_feature_labels(self):
        
        atom_feature_labels = []
        
        for characteristic in self.atom_allowable_sets.keys():
            if characteristic in self.NON_ONE_HOT_ENCODED:
                atom_feature_labels.append(characteristic)
            else:
                atom_feature_labels += [f"{characteristic}_{value}" for value in self.atom_allowable_sets[characteristic]]
        
        return atom_feature_labels
    
    
    def get_bond_feature_labels(self):
        
        bond_feature_labels = []
        
        for characteristic in self.bond_allowable_sets.keys():
            if characteristic in self.NON_ONE_HOT_ENCODED:
                bond_feature_labels.append(characteristic)
            else:
                bond_feature_labels += [f"{characteristic}_{value}" for value in self.bond_allowable_sets[characteristic]]
        
        return bond_feature_labels
    

    def get_default_atom_allowable_set(self, atom_characteristic):
        match atom_characteristic:
            case "symbol":
                return ['C', 'O', 'N', 'Cl', 'S', 'F', 'Na', 'P', 'Br', 'Si', 'K', 'Sn', 'UNK'],
            case "degree":
                return [2, 1, 3, 4, 0]
            case "hybridization":
                return [Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3,
                        Chem.rdchem.HybridizationType.S,
                        Chem.rdchem.HybridizationType.SP]
            case "total_num_hs":
                return [0, 1, 2, 3]
            case "implicit_valence":
                return [0, 1, 2, 3, 4]
            case "explicit_valence":
                return [4, 2, 3, 1, 0, 'UNK']
            case "formal_charge":
                return [0, -1, 1, 'UNK']
            case _:
                raise ValueError(f"No default allowable set for atom characteristic '{atom_characteristic}'. You must set your own allowable set.")

    
    def get_default_bond_allowable_set(self, bond_characteristic):
        match bond_characteristic:
            case "bond_type":
                return [Chem.rdchem.BondType.SINGLE,
                        Chem.rdchem.BondType.DOUBLE,
                        Chem.rdchem.BondType.TRIPLE,
                        Chem.rdchem.BondType.AROMATIC]
            case _:
                raise ValueError(f"No default allowable set for bond characteristic '{bond_characteristic}'. You must set your own allowable set.")
        

    def set_default_config(self):

        atom_allowable_sets = {
            "symbol": self.get_default_atom_allowable_set("symbol"),
            "degree": self.get_default_atom_allowable_set("degree"),
            "hybridization": self.get_default_atom_allowable_set("hybridization"),
            "total_num_hs": self.get_default_atom_allowable_set("total_num_hs"),
            "implicit_valence": self.get_default_atom_allowable_set("implicit_valence"),
            "explicit_valence": self.get_default_atom_allowable_set("explicit_valence"),
            "is_aromatic": None,
            "_ChiralityPossible": None,
            "formal_charge": self.get_default_atom_allowable_set("formal_charge"),
        }

        bond_allowable_sets = {
            "bond_type": self.get_default_bond_allowable_set("bond_type"),
            "is_conjugated": None,
            "is_in_ring": None,
        }


        self.set_atom_allowable_sets(atom_allowable_sets)

        if self.include_edge_features:
            self.set_bond_allowable_sets(bond_allowable_sets)
    
    
    def extract_molecular_features(self, mol):
        
        mol_atom_features = []
        for atom in mol.GetAtoms():
            mol_atom_features.append(self.atom_features(atom))
        mol_atom_features = torch.tensor(mol_atom_features, dtype=torch.float32)
        
        if not self.include_edge_features:
            return mol_atom_features, None
        
        mol_bond_features = []
        for bond in mol.GetBonds():
            mol_bond_features.append(self.bond_features(bond))
            mol_bond_features.append(self.bond_features(bond)) # do twice (undirectional graph)
        mol_bond_features = torch.tensor(mol_bond_features, dtype=torch.float32)
        
        return mol_atom_features, mol_bond_features
        
        
    def atom_features(self, atom):
        
        feats = []
        for characteristic in self.atom_allowable_sets.keys():
            property_getter = self.SUPPORTED_ATOM_CHARACTERISTICS[characteristic]
            feat = property_getter(atom)
            
            if characteristic in self.NON_ONE_HOT_ENCODED:
                feats.append(feat)
            else:
                allowable_set = self.atom_allowable_sets[characteristic]
                if 'UNK' in allowable_set:
                    one_hot_encoded_feat = self.one_of_k_encoding_unk(feat, allowable_set)
                else:
                    one_hot_encoded_feat = self.one_of_k_encoding(feat, allowable_set, characteristic_name=characteristic)
                feats.extend(one_hot_encoded_feat)
        return feats
    

    def bond_features(self, bond):

        feats = []
        for characteristic in self.bond_allowable_sets.keys():
            property_getter = self.SUPPORTED_BOND_CHARACTERISTICS[characteristic]
            feat = property_getter(bond)
            
            if characteristic in self.NON_ONE_HOT_ENCODED:
                feats.append(feat)
            else:
                allowable_set = self.bond_allowable_sets[characteristic]
                if 'UNK' in allowable_set:
                    one_hot_encoded_feat = self.one_of_k_encoding_unk(feat, allowable_set)
                else:
                    one_hot_encoded_feat = self.one_of_k_encoding(feat, allowable_set, characteristic_name=characteristic)
                feats.extend(one_hot_encoded_feat)
        return feats

                
    def one_of_k_encoding(self, x, allowable_set, characteristic_name=None):
        if x not in allowable_set:
            characteristic_text = f"{characteristic_name} " if characteristic_name else ""
            self.warning(f"Ignoring input {characteristic_text}{x}, not in allowable set {allowable_set}")
        return [x == s for s in allowable_set]


    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to UNK."""
        if x not in allowable_set:
            x = 'UNK'
        return [x == s for s in allowable_set]
    
    
    def adjacency_matrix(self, mol):
    
        ix1, ix2 = [], []
        for bond in mol.GetBonds():

            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ix1 += [start, end]
            ix2 += [end, start]

        adj_norm = torch.asarray([ix1, ix2], dtype=torch.int64) # Needs to be in COO Format
        return adj_norm
        
        
    def get_num_node_features(self):
        return len(self.get_atom_feature_labels())
    

    def get_num_edge_features(self):
        return len(self.get_bond_feature_labels())
    
    
    def warning(self, message):
        if self.warnings_enabled:
            warnings.warn(message)
    
    def enable_warnings(self, enable=True):
        self.warnings_enabled = enable
    
    
    def featurize(self, sm, y=None):
        
        mol = Chem.MolFromSmiles(sm)
        adjacency_matrix = self.adjacency_matrix(mol)
        mol_atom_features, mol_bond_features = self.extract_molecular_features(mol)
        
        return Data(x=mol_atom_features,
                    edge_index=adjacency_matrix,
                    edge_attr=mol_bond_features,
                    y=y,
                    smiles=sm)
        
    
    def save_config(self, config_file="featurizer_config.pkl"):
        config = {
            'warnings_enabled': self.warnings_enabled,
            'include_edge_features': self.include_edge_features,
            'atom_allowable_sets': self.atom_allowable_sets,
            'bond_allowable_sets': self.bond_allowable_sets,
        }

        with open(config_file, 'wb') as f:
            pickle.dump(config, f)

        
    def load_config(self, config_file):

        with open(config_file, 'rb') as f:
            config = pickle.load(f)

        self.warnings_enabled = config['warnings_enabled']
        self.include_edge_features = config['include_edge_features']

        self.set_atom_allowable_sets(config['atom_allowable_sets'])

        self.set_bond_allowable_sets(config['bond_allowable_sets'])

        return self
    

    def save(self, filepath='featurizer.pkl'):
        with open(filepath, "wb") as file:
            pickle.dump(self, file)
    

    def __call__(self, sm, y=None):
        return self.featurize(sm, y)
    

    def __repr__(self):
        attributes = {
            'atom_allowable_sets': self.atom_allowable_sets,
            'bond_allowable_sets': self.bond_allowable_sets,
        }
        return json.dumps(attributes, indent=4)
    
    def __str__(self):
        return self.__repr__()


class SmilesGraphDataset(Dataset):

    def __init__(self, smiles, y=None, featurizer=None):
        super(SmilesGraphDataset, self).__init__()
        
        self.smiles = smiles
        self.y = y
        
        if featurizer:
            self.featurizer = featurizer
        else:
            self.featurizer = SmilesGraphFeaturizer()
            self.featurizer.set_default_config()
        
        self.precomputed_features = None
    
    
    def config_from_other_dataset(self, dataset):
        self.featurizer = SmilesGraphFeaturizer().config_from_other_featurizer(dataset.featurizer)
        return self

    
    def get_atom_feature_labels(self):        
        return self.featurizer.get_atom_feature_labels()
    
    
    def get_bond_feature_labels(self):
        return self.featurizer.get_bond_feature_labels()
        
        
    def precompute_featurization(self):
        if self.y:
            self.precomputed_features = [self.featurizer(sm, y) for sm, y, in zip(self.smiles, self.y)]
        else:
            self.precomputed_features = [self.featurizer(sm) for sm in self.smiles]

        
    def get_num_node_features(self):
        return len(self.get_atom_feature_labels())
    
    
    def get_num_edge_features(self):
        return len(self.get_bond_feature_labels())

    
    def __getitem__(self, idx):
        
        if self.precomputed_features:
            return self.precomputed_features[idx]
        
        sm = self.smiles[idx]
        y = self.y[idx] if self.y else None

        
        return self.featurizer(sm, y)

    
    def __len__(self):
        return len(self.smiles)
    
