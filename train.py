# imports

from data.fwf_dataset import FwfDataset
from models.fgf import FGFeatNetwork
from utils.metrics import get_multilevel_metrics, print_metrics, combine_metrics_list, EarlyStopping
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
# from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy

def main():
    # args
    parser = ArgumentParser()
    parser.add_argument('-c','--config', default='./config/default.yaml')
    args = parser.parse_args()

    # assemble config
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.data.split))
    with open(os.path.join(cfg.data.dataset_root, 'class_dict.json'),'r') as f:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({'data':{'label_schema':json.load(f)}}))


    print(OmegaConf.to_yaml(cfg))
    exp_dir = f"./exp/{generate_timestamp()}_{cfg.experiment.title_suffix}"
    os.makedirs(exp_dir,exist_ok=True)
    OmegaConf.save(cfg, f=os.path.join(exp_dir,'config.yaml'))


    # initialize datasets
    train_ds = FwfDataset(cfg, cfg.data.preprocessing._transformsTraining_, cfg.data._trainProjects_)
    val_ds = FwfDataset(cfg, cfg.data.preprocessing._transformsValidation_, cfg.data._valProjects_)

    # pre-computations: normals, inc angles, etc.
    train_ds.subsample_grid(cfg.data.query_grid_size, save_inv=True)
    train_ds.compute_neibors_knn(k=cfg.data.num_neib_normalsComputation)
    train_ds.compute_normals_knn()
    train_ds.compute_incAngles()


    val_ds.subsample_grid(cfg.data.query_grid_size, save_inv=True)
    val_ds.compute_neibors_knn(k=cfg.data.num_neib_normalsComputation)
    val_ds.compute_normals_knn()
    val_ds.compute_incAngles()
    val_ds.compute_neibors_knn(k=cfg.data.num_neib_featureExtraction)

    # load class weights
    if cfg.data.weighted_random_subsample:
        train_ds.load_class_weights()
        val_ds.load_class_weights()

    

    # create model
    model = FGFeatNetwork(cfg=cfg,
        num_input_feats = train_ds[0]['features_neibors'].shape[-1],
        ).to(device=cfg.general.device)





    criterion = CrossEntropyLoss()
    optim = Adam(params=model.parameters(), weight_decay=cfg.general.weight_decay)
    val_dl = DataLoader(val_ds, batch_size=cfg.general.batch_size, num_workers = cfg.general.num_workers)
    early_stopping = EarlyStopping(min_delta=cfg.general.early_stopping_minDelta)
  
    
    # metrics
    miou_metric = {label_level:MulticlassJaccardIndex(num_classes = \
        len(cfg.data.label_schema[label_level]),average='macro').to(device=cfg.general.device)\
             for label_level in cfg.data.label_names}
    macc_metric = {label_level:MulticlassAccuracy(num_classes = \
        len(cfg.data.label_schema[label_level]),average='macro').to(device=cfg.general.device)\
             for label_level in cfg.data.label_names}
    
    # train loop
    for epoch in range(cfg.general.max_epochs):
        print(f"Epoch-{epoch:03}")

        
        # train
        
        # subsample a random percentage of points
        start = time()
        print(f'Sampling training dataset.',end=' ')
        train_ds.subsample_random(0.01,weighted=cfg.data.weighted_random_subsample)
        train_ds.compute_neibors_knn(k=cfg.data.num_neib_featureExtraction, verbose=False)
        train_dl = DataLoader(train_ds, batch_size=cfg.general.batch_size, num_workers = cfg.general.num_workers, pin_memory=True)
        print(f'Done. Took {time()-start:.2f}.')
        
        # containers
        pred_container = {label_level : torch.ones(len(train_ds))*-1 for label_level in cfg.data.label_names}
        gt_container = {label_level : torch.ones(len(train_ds))*-1 for label_level in cfg.data.label_names}
        epoch_train_loss = []

        model.train()
        for batch_i, batch in enumerate(tqdm(train_dl, desc=f"{'Training':<15}", leave=True)):
            optim.zero_grad()
            # put batch on device
            for k, v in batch.items():
                batch[k] = v.to(device=cfg.general.device)

            # forward pass
            out = model(batch)
            
            # agregate loss on all levels
            loss = torch.tensor(0.).to(device=cfg.general.device)
            for k in out.keys():
                output = out[k]
                gt = batch[k] # type:ignore
                loss += criterion(output, gt)
            
            loss.backward()
            optim.step()


            # aggregate values for metric calculation
            epoch_train_loss.append(loss.item())
            preds = {k:torch.argmax(v,dim=1) for k,v in out.items()}
            for label_name in cfg.data.label_names:
                # fill containers
                pred_container[label_name][batch_i * cfg.general.batch_size : (batch_i+1) * cfg.general.batch_size,...] = preds[label_name]
                gt_container[label_name][batch_i * cfg.general.batch_size : (batch_i+1) * cfg.general.batch_size,...] = batch[label_name]
            
            

            
            del batch, out
            torch.cuda.empty_cache()
        
        # epoch metrics
        print(f"Training. Loss{np.mean(epoch_train_loss):.4f};")
        for label_level in cfg.data.label_names:
            assert torch.all(pred_container[label_name] >= 0) # ensure that all entries in the dataset are filled
            assert torch.all(gt_container[label_name] >= 0) # ensure that all entries in the dataset are filled
            miou = miou_metric[label_level](gt_container[label_name], pred_container[label_name]).item()
            macc = macc_metric[label_level](gt_container[label_name], pred_container[label_name]).item()
            print(f'{label_level}-> mIoU: {miou*100:.2f}; mAcc: {macc*100:.2f}')
        
        # validate  
        epoch_metrics = []
        epoch_val_loss = []
        iou_sklearn = {label_level:[] for label_level in val_ds.label_names}
        iou_torchmetrics = {label_level:[] for label_level in val_ds.label_names}
        model.eval()
        with torch.no_grad():
            for batch_i,batch in enumerate(tqdm(val_dl, desc=f"{'Validation':<15}", leave=True)):
                # put batch on device
                for k, v in batch.items():
                    batch[k] = v.to(device=cfg.general.device)

                # forward pass
                out = model(batch)
                # agregate loss on all levels
                loss = torch.tensor(0.).to(device=cfg.general.device)
                for k in out.keys():
                    output = out[k]
                    gt = batch[k] # type:ignore
                    loss += criterion(output, gt)

            # aggregate values for metric calculation
            epoch_val_loss.append(loss.item())
            preds = {k:torch.argmax(v,dim=1) for k,v in out.items()}
            
            
            # OLD Metrics
            # epoch_metrics.append(get_multilevel_metrics(preds, batch, cfg))
            # # sklearn
            # for label_level in train_ds.label_names:
            #     iou_sklearn[label_level].append(jaccard_score(
            #         batch[label_level].detach().cpu().numpy(),
            #         preds[label_level].detach().cpu().numpy(),average='micro'))
            # # torchmetrics
            # for label_level in train_ds.label_names:
            #     miou = MeanIoU(num_classes=len(cfg.data.label_schema[label_level]),input_format='index', per_class=True).to(device=cfg.general.device)
            #     iou_torchmetrics[label_level].append(miou(
            #         preds[label_level][:,None].to(dtype=torch.int64),
            #         batch[label_level][:,None].to(dtype=torch.int64)))
            
            del batch, out
            torch.cuda.empty_cache()

                

        # average out the epoch metrics
        # own metrics
        epoch_metrics = combine_metrics_list(epoch_metrics,cfg)
        print('own metrics:')
        print_metrics(epoch_metrics, cfg)
        # sklearn
        iou_sklearn = {k:np.mean(v) for k,v in iou_sklearn.items()}
        print(f'sklearn miou: {iou_sklearn}')
        #torchmetrics
        iou_torchmetrics = {k:torch.cat([vv[None,:] for vv in v],dim=0).mean(dim=0) for k,v in iou_torchmetrics.items()}
        for k,v in iou_torchmetrics.items():
            print(f'{k} : {v}')

        print(f"L_val:{np.mean(epoch_val_loss):.4f}")
        early_stopping(np.mean(epoch_val_loss))
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        torch.save(model.state_dict(),os.path.join(exp_dir,'latest_model.pth'))
        if early_stopping.new_best:
            torch.save(model.state_dict(),os.path.join(exp_dir,'best_model.pth'))
        
    print('\n')


if __name__=='__main__':
    main()


