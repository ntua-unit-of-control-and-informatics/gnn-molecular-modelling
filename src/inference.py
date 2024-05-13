import random
import numpy as np
from pathlib import Path
import logging
import sys
from dataloaders import endpoint_target_mean_std
from utilities import StandardNormalizer, check_gpu_availability
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import csv

import pickle


from dataloaders import SmilesGraphDataset


def inference(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    # args.load_model_dir = Path(args.load_model_dir).resolve()

    # load_model_filepath = args.load_model_dir/'model_scripted.pt'
    

    working_dir = Path.cwd()
    filename = f'{args.endpoint_name}_dataset.csv'
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        data_dir = working_dir.parent/'data'/args.endpoint_name
    dataset_filepath = data_dir/filename

    args.inference_output_dir = working_dir
    output_filepath = args.inference_output_dir/'inference.csv'
    

    args.load_model_dir = working_dir.parent/'experiments'/'hc20'/'Model_355'
    load_model_filepath = args.load_model_dir/'model_scripted.pt'

    print(dataset_filepath)
    print(load_model_filepath)
    # exit()



    handlers=[logging.StreamHandler(sys.stdout)]

    # info logger for saving command line outputs during training
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)

    logging.info('Inference Mode\n')

    # matters only for regression
    if args.task == 'regression' and args.normalize_target:
        target_mean, target_std = endpoint_target_mean_std(args.endpoint_name)
        target_normalizer = StandardNormalizer(target_mean, target_std)
    else:
        target_normalizer = None
    

    # Device
    device = torch.device('cuda:0' if not check_gpu_availability(not args.no_cuda) else 'cpu')

    torch.manual_seed(args.seed)
    if check_gpu_availability(not args.no_cuda):
        logging.info(f"\nDevice: \n- {torch.cuda.get_device_name()}")
        torch.cuda.manual_seed(args.seed)
    else:
        logging.info(f"\nDevice: {'CPU'}")



    model = torch.jit.load(load_model_filepath).to(device)
    print(model)

    try:
        featurizer_filepath = args.load_model_dir/'featurizer.pkl'
        with open(featurizer_filepath, "rb") as f:
            featurizer = pickle.load(f)
    except Exception as e:
        print(e)
        print('TODO Featurizer Exception')

    # try:
    #     doa_filepath = args.load_model_dir/'doa.pkl'
    #     with open(doa_filepath, "rb") as f:
    #         doa = pickle.load(f)
    # except Exception as e:
    #     print(e)
    #     print('TODO DoA Exception')

    print(featurizer)

    
    
    
    
    df = pd.read_csv(dataset_filepath)
    smiles = df[['SMILES']].values.ravel().tolist()
    print(len(smiles))
    

    dataset = SmilesGraphDataset(smiles, featurizer=featurizer)

    dataset.precompute_featurization()
    print(featurizer.get_atom_feature_labels())


    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if check_gpu_availability(not args.no_cuda) else {}
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    
    if args.task=='binary':
        binary_decision_threshold = 0.5 # maybe let this be in args
        preds = predict_binary(loader, model, device, args.endpoint_name, decision_threshold=binary_decision_threshold, return_probs=False)
    elif args.task=='regression':
        preds = predict_regression(loader, model, device, args.endpoint_name, target_normalizer=target_normalizer, output_filepath=output_filepath)
    else:
        raise ValueError(f"Unsupported task type '{args.task}'")



    # print(preds)
    [
        model,
        loader,
        device,
        featurizer,
        # doa,
    ]





def predict_binary(loader, model, device, endpoint_name, decision_threshold=0.5, return_probs=False, output_filepath=None):

    all_preds = []
    all_probs = []
    
    if output_filepath is not None:
        f = open(output_filepath, mode='w', newline='')
        writer = csv.writer(f)
        writer.writerow(['SMILES', endpoint_name])

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(loader):
            
            data = data.to(device)
            smiles = data.smiles
            
            outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch).squeeze(-1)
            
            probs = F.sigmoid(outputs)
            preds = (probs > decision_threshold).int()
            
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

            if output_filepath is not None:
                for sm, pred in zip(smiles, preds):
                    writer.writerow([sm, pred])
    
    if output_filepath is not None:
        f.close()

    if return_probs:    
        return all_preds, all_probs
    else:
        return all_preds


def predict_regression(loader, model, device, endpoint_name, target_normalizer=None, output_filepath=None):
    
    all_preds = []

    if output_filepath is not None:
        f = open(output_filepath, mode='w', newline='')
        writer = csv.writer(f)
        writer.writerow(['SMILES', endpoint_name])

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(loader):

            data = data.to(device)
            smiles = data.smiles
            
            outputs = model(x=data.x, edge_index=data.edge_index, batch=data.batch).squeeze(-1) # .cpu()
            
            all_preds.extend(outputs.tolist())

            if target_normalizer:
                preds = target_normalizer.denormalize(outputs).tolist()
            else:
                preds = outputs.tolist()

            if output_filepath is not None:
                for sm, pred in zip(smiles, preds):
                    writer.writerow([sm, pred])

    if output_filepath is not None:
        f.close()

    if target_normalizer:
        all_preds = target_normalizer.denormalize(torch.tensor(all_preds)).tolist()

    return all_preds 

