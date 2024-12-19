from data.fwf_dataset import FwfDataset
from models.fgf import FGFeatNetwork
from utils.metrics import get_multilevel_metrics, print_metrics, combine_metrics_list, EarlyStopping, simple_metrics
from utils.general import generate_timestamp

from argparse import ArgumentParser
from tqdm import tqdm
from time import time
import numpy as np
import json
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from omegaconf import OmegaConf

# metrics computation alternatives
from sklearn.metrics import jaccard_score
from torchmetrics.segmentation import MeanIoU

import pandas as pd
from plyfile import PlyData, PlyElement
from torch.nn.functional import softmax
import pickle






def main():





    parser = ArgumentParser()
    parser.add_argument('-e', '--exp_dir',help='Directory with the model .pth file and the training config, e.g. ./exp/2024-12-16_16-04-25_v006')
    parser.add_argument('-m', '--model_version', default='latest_model.pth')
    parser.add_argument('-v', '--voxel_size', type=float, default=0.1, help='voxel size with which to sample the point cloud')
    parser.add_argument('--return_probs', action='store_true')
    parser.add_argument('--return_embeddings', action='store_true')
    parser.add_argument('--test_on_train', action='store_true')

    args = parser.parse_args()


    output_dir = os.path.join(args.exp_dir,'test' if not args.test_on_train else 'test_on_train')

    # build config

    cfg = OmegaConf.load(os.path.join(args.exp_dir,'config.yaml'))
    with open(os.path.join(cfg.data.dataset_root, 'class_dict.json'),'r') as f:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({'data':{'label_schema':json.load(f)}}))

    if 'embedding_size' not in cfg.model.keys():
        cfg.model.embedding_size = 128
    print(OmegaConf.to_yaml(cfg))



    report = {} # project, label_level, overall, classwise
    project_names = cfg.data._testProjects_ if not args.test_on_train else cfg.data._trainProjects_
    for test_project in project_names:
        test_ds = FwfDataset(cfg, cfg.data.preprocessing._transformsValidation_, [test_project],
                            return_resiIdx=True, return_projIdx=True)
        proj_name = test_ds.projects[0]['proj_name']
        print(proj_name)
        
        test_ds.subsample_grid(args.voxel_size, save_inv=True)
        test_ds.compute_neibors_knn(k=cfg.data.num_neib_normalsComputation)
        test_ds.compute_normals_knn()
        test_ds.compute_incAngles()
        test_ds.compute_neibors_knn(k=cfg.data.num_neib_featureExtraction, verbose=True)
        model = FGFeatNetwork(cfg=cfg,
            num_input_feats = test_ds[0]['features_neibors'].shape[-1],
            ).to(device=cfg.general.device)
        model.load_state_dict(torch.load(os.path.join(args.exp_dir,args.model_version), weights_only=True))
        
        test_dl = DataLoader(test_ds, batch_size=cfg.testing.batch_size, num_workers=cfg.general.num_workers)

        

        # initiate containers
        all_preds = [dict() for p in test_ds.projects]

        if args.return_probs:
            all_probs = [dict() for p in test_ds.projects]


        for pi, proj in enumerate(test_ds.projects):
            for label_name in test_ds.label_names:

                all_preds[pi][label_name] = torch.ones(size = (test_ds.projects[pi]['xyz_sub'].shape[0],)).to(
                    device=cfg.general.device,dtype=torch.int64)*-1
                if args.return_probs:
                    all_probs[pi][label_name] = torch.ones(size = (test_ds.projects[pi]['xyz_sub'].shape[0],
                        len(cfg.data.label_schema[label_name]))).to(device=cfg.general.device,dtype=torch.float32)*-1

        if args.return_embeddings:                
            embeddings = list()
            for pi, proj in enumerate(test_ds.projects):
                embeddings.append(torch.zeros(size=(test_ds.projects[pi]['xyz_sub'].shape[0],cfg.model.embedding_size)).to(device=cfg.general.device))

        # run test loop
        with torch.no_grad():
            for batch in tqdm(test_dl):
                # put batch on device
                for k, v in batch.items():
                    batch[k] = v.to(device=cfg.general.device)
                if args.return_embeddings:
                    out = model(batch, return_embedding=True)
                else:
                    out = model(batch)
                probs = dict()
                for k,v in [(k,v) for k,v in out.items() if k != 'embedding']:
                    probs[k] = softmax(v,dim=-1)
                    preds = torch.argmax(probs[k], dim=-1)
                    all_preds[0][k][batch['residual_idx']] = preds # assume dataset has one project inside !
                    if args.return_probs:
                        all_probs[0][k][batch['residual_idx']] = probs[k]
                    if args.return_embeddings:
                        embeddings[0][batch['residual_idx']] = out['embedding']
            # convert to numpy and upsample
            for k in cfg.data.label_names:
                all_preds[0][k] = all_preds[0][k].cpu().detach().numpy()
                all_preds[0][k] = all_preds[0][k][test_ds.projects[0]['sub_inv']]
                if args.return_probs:
                    all_probs[0][k] = all_probs[0][k].cpu().detach().numpy()
                    all_probs[0][k] = all_probs[0][k][test_ds.projects[0]['sub_inv']]

        

        # map dataset field names to single columns
        column_names_from_fields = {
            'xyz' : ['x','y','z'], 
            'rgb' : ['Red', 'Green','Blue'], 
            'riegl_feats' : ['riegl_reflectance','riegl_amplitude', 'riegl_deviation', 'riegl_targetIndex','riegl_targetCount'], 
            'geom_feats' : ['linearity','planarity','sphericity','omnivariance','anisotropy','eigenentropy',], 
            'normals' : ['nx','ny','nz'], 
            'incAngles' : ['incAngles'], 
            'distanceFromScanner' : ['distanceFromScanner'], 
        }
        # undo normalization
        test_ds.projects[0]['rgb'] *= np.array([[0.21558372, 0.23351644, 0.21213871]])
        test_ds.projects[0]['rgb'] += np.array([[0.29689665, 0.3428666,  0.190237]])
        test_ds.projects[0]['riegl_feats'] *= np.array([[  2.32590898, 2.98518547, 929.71399545, 1., 0.22651793]])
        test_ds.projects[0]['riegl_feats'] += np.array([[-6.90191949, 25.16398933, 26.45952891,  1., 1.03636612]])
        out_data = [test_ds.projects[0][k] for k in cfg.data.scalar_input_fields]
        out_column_names = [s for ss in [column_names_from_fields[f] for f in cfg.data.scalar_input_fields] for s in ss]
        
        # get ground_truth labels
        gt_names = [f"gt_{g}" for g in test_ds.label_names]
        gt_labels = test_ds.projects[0]['labels']
        out_data.append(gt_labels)
        out_column_names += gt_names
        
        # predictions
        pr_names = [f"pr_{g}" for g in test_ds.label_names]
        pr_labels = [all_preds[0][g][:,None] for g in test_ds.label_names]
        out_data += pr_labels
        out_column_names += pr_names

        # probabilities
        if args.return_probs:
            prob_names = [i for ii in [cfg.data.label_schema[level] for level in test_ds.label_names] for i in ii] # FIXME: Only works for single levels at a time
            prob_data = [all_probs[0][g] for g in test_ds.label_names]
            out_data += prob_data
            out_column_names += prob_names


        # break
        pcd = pd.DataFrame(np.concatenate(out_data,axis=-1), columns=out_column_names)
        os.makedirs(output_dir, exist_ok=True)
        PlyData([PlyElement.describe(pcd.to_records(index=False),'vertex')]).write(os.path.join(output_dir,proj_name.replace("::","--")+".ply"))

        # save embeddings + inverse indices
        # TODO:
        
        
        # metrics
        # FIXME: shapes, indices, etc. will break for more projects and or more than one label levels
        report[proj_name] = dict()
        for li, l in enumerate(test_ds.label_names):
            metrics = simple_metrics(np.squeeze(gt_labels), np.squeeze(pr_labels), len(cfg.data.label_schema[l]))
            report[proj_name][l] = metrics
        
        # overwrite report
        # report_name = 'report.pkl' if not args.test_on_train else 'report_on_train.pkl'
        with open(os.path.join(output_dir,'report.pkl'),'wb') as f:
            pickle.dump(report,f)

if __name__ == '__main__':
    main()