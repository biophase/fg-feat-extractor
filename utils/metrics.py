from typing import List, Dict
import torch
from torch import nn
import pandas as pd
import numpy as np


def get_multilevel_weights(dataset, cfg):
    weights = {level: None for level in cfg.data.label_names}
    for level in cfg.data.label_names:
        weights[level] = torch.zeros(len(dataset.schema.schema[level]))
        dataset.label_counts[level]
        total_counts = sum(dataset.label_counts[level].values())
        num_classes = len(dataset.schema.schema[level])
        print(f"level: {level}, total_counts: {total_counts}, num_classes: {num_classes}")
        for i, (label_name, label_count) in enumerate(dataset.label_counts[level].items()):
            ii = dataset.schema.schema[level][label_name]
            # assert ii == i, f"Index mismatch: {ii} != {i} for level {level}"
            weights[level][ii] = total_counts/(num_classes * label_count + 1e-6)
            # weights[level] /= weights[level].sum()
            weights[level] = weights[level].to(device=cfg.general.device)
            # weights[level] = torch.tensor(weights[level], dtype=torch.float32, device=cfg.general.device)
            
    return weights



def get_multilevel_criterion(weights, cfg):
    criterion = dict()
    for level in cfg.data.label_names:
        if cfg.train.use_class_weights:
            criterion[level] = nn.CrossEntropyLoss(weight=weights[level])
        else:
            criterion[level] = nn.CrossEntropyLoss()
    return criterion


def multilevel_ce_loss(criterion, y_pred, y_true, cfg):
    loss = 0
    for level, level_weight in zip(cfg.data.label_names, cfg.train.label_levels_weights):
        a = y_pred[str(level)]
        b = y_true[:,level].long()
        level_weight = torch.tensor(level_weight).to(cfg.general.device)
        loss += level_weight * criterion[level](a, b)
    return loss

def print_metrics(metrics, cfg, print_iou= False):

    metrics_names = ['oA','mIoU','mP','mR','mF1','mcAcc']
    metrics_df = pd.DataFrame(columns=metrics_names)
    
    for level in cfg.data.label_names:
        
        metric_means = [metrics[level]['overall']['overall_accuracy']]

        if print_iou:
            print(f"IoU at Level: {level}")        
            for cls_name, iou in zip(cfg.data.label_schema[level].keys(),metrics[level]['classwise']['iou']):
                print(f"\t{cls_name:16}: {iou:.3f}")

        for metric in ['iou','precision','recall','f1_score','class_acc']:
            metric_means.append(np.nanmean(metrics[level]['classwise'][metric]))    
            # print(f"\t{metric:16}: {np.nanmean(metrics[level]['classwise'][metric])}")
        metrics_df.loc[level] = metric_means
    
    print(metrics_df)
    
def get_multilevel_metrics(epoch_predictions, epoch_labels, cfg):
    with torch.no_grad():
        metrics = {level:dict() for level in cfg.data.label_names}
        for level in cfg.data.label_names:
            num_classes_at_level = len(cfg.data.label_schema[level])

            metrics[level]['overall']=dict()
            metrics[level]['classwise']=dict()

            y_pred = epoch_predictions[level]
            y_true = epoch_labels[level]
            k_at_level = torch.arange(num_classes_at_level)[None,:].to(cfg.general.device)
            true_equals_k = k_at_level == y_true[:,None]
            pred_equals_k = k_at_level == y_pred[:,None]

            # initialize tensors to store the metrics
            iou = torch.zeros((num_classes_at_level)).to(cfg.general.device)
            precision = torch.zeros((num_classes_at_level)).to(cfg.general.device)
            recall = torch.zeros((num_classes_at_level)).to(cfg.general.device)
            f1_score = torch.zeros((num_classes_at_level)).to(cfg.general.device)
            class_acc = torch.zeros((num_classes_at_level)).to(cfg.general.device)
            
            # calculate class-wise metrics
            for k in range(num_classes_at_level):
                tp = true_equals_k[:,k] & pred_equals_k[:,k]
                fn = true_equals_k[:,k] & ~pred_equals_k[:,k]
                fp = ~true_equals_k[:,k] & pred_equals_k[:,k]

                iou[k] = tp.sum() / ((tp.sum() + fn.sum() + fp.sum()) + 1e-6)
                precision[k] = tp.sum() / ((tp.sum() + fp.sum()) + 1e-6)
                recall[k] = tp.sum() / ((tp.sum() + fn.sum()) + 1e-6)
                f1_score[k] = 2 * precision[k] * recall[k] / ((precision[k] + recall[k]) + 1e-6)
                class_acc[k] = tp.sum() / (true_equals_k[:,k].sum() + 1e-6)

            # calculate overall metrics
            overall_accuracy = torch.eq(y_pred, y_true).float().mean().item()
            confusion_matrix = torch.zeros((num_classes_at_level, num_classes_at_level)).to(device=cfg.general.device)
            
            # calculate confusion matrix
            true_labels = torch.argmax(true_equals_k.short(),dim=1)
            pred_labels = torch.argmax(pred_equals_k.short(),dim=1)
            values = torch.ones(len(true_labels)).to(device=cfg.general.device)
            confusion_matrix.index_put_((true_labels,pred_labels), values, accumulate=True)


            # save values in a dictionary
            metrics[level]['overall']['overall_accuracy'] = overall_accuracy
            metrics[level]['overall']['confusion_matrix'] = confusion_matrix.detach().cpu().numpy()

            metrics[level]['classwise']['iou'] = iou.detach().cpu().numpy()
            metrics[level]['classwise']['precision'] = precision.detach().cpu().numpy()
            metrics[level]['classwise']['recall'] = recall.detach().cpu().numpy()
            metrics[level]['classwise']['f1_score'] = f1_score.detach().cpu().numpy()
            metrics[level]['classwise']['class_acc'] = class_acc.detach().cpu().numpy()

        return metrics



def combine_metrics_list(metrics_list:List[Dict], cfg):

    metrics = {ln : dict() for ln in cfg.data.label_names}
    for label_name in cfg.data.label_names:
        # classwise
        metrics[label_name]['classwise'] = {metric: np.concatenate([metrics_list[e][label_name]['classwise'][metric][None,:] 
                                                                    for e in range(len(metrics_list))],axis=0).mean(axis=0) 
                                                                    for metric in metrics_list[0][label_name]['classwise'].keys()}
        # overall
        metrics[label_name]['overall'] = dict()
        metrics[label_name]['overall']['overall_accuracy'] = np.mean([metrics_list[e][label_name]['overall']['overall_accuracy'] for e in range(len(metrics_list))])
        metrics[label_name]['overall']['confusion_matrix'] = np.concatenate([metrics_list[e][label_name]['overall']['confusion_matrix'][None,...] for e in range(len(metrics_list))],axis=0).sum(axis=0)
    return metrics


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.new_best = False

    def __call__(self, value):
        self.new_best = False
        if self.best_value is None:
            self.best_value = value
        elif value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
            self.new_best = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
def simple_metrics(gt:np.ndarray, pr:np.ndarray, num_classes:int)->Dict:
    """
    gt -> ground truth (N,)
    pr -> preds (N,)
    returns dict with 
        - iou (num_classes,)
        - pc_acc (num_classes,)
        - miou (1,) 
        - macc (1,)
    """

    assert len(gt.shape) == 1 
    assert len(pr.shape) == 1 
    assert gt.shape[0] == pr.shape[0]

    c = np.arange(num_classes)
    gt_eq_c = gt[:,None] == c [None,:]
    pr_eq_c = pr[:,None] == c [None,:]

    iou = np.zeros((num_classes))
    pc_acc = np.zeros((num_classes))
    for ci in range(num_classes):
        tp = (gt_eq_c[:,ci]==True) & (pr_eq_c[:,ci]==True)
        fn = (gt_eq_c[:,ci]==True) & (pr_eq_c[:,ci]==False)
        fp = (gt_eq_c[:,ci]==False) & (pr_eq_c[:,ci]==True)
        tp_plus_fn = fp.sum() + fn.sum()
        iou[ci] = tp.sum() / (tp_plus_fn + tp.sum() +1e-6)
        pc_acc[ci] = tp.sum() / tp_plus_fn if tp_plus_fn > 0 else np.nan



    # put nans if there were no gt to begin with
    iou[gt_eq_c.sum(axis=0) == 0]=np.nan
    miou = np.nanmean(iou)
    macc = np.nanmean(pc_acc)
    return dict(
        iou = iou, 
        pc_acc = pc_acc,
        miou = miou,
        macc = macc
    )