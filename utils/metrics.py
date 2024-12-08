import torch
from torch import nn
import pandas as pd
import numpy as np


def get_multilevel_weights(dataset, cfg):
    weights = {level: None for level in cfg.train.train_on_label_levels}
    for level in cfg.train.train_on_label_levels:
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
            weights[level] = weights[level].to(device=cfg.train.device)
            # weights[level] = torch.tensor(weights[level], dtype=torch.float32, device=cfg.train.device)
            
    return weights



def get_multilevel_criterion(weights, cfg):
    criterion = dict()
    for level in cfg.train.train_on_label_levels:
        if cfg.train.use_class_weights:
            criterion[level] = nn.CrossEntropyLoss(weight=weights[level])
        else:
            criterion[level] = nn.CrossEntropyLoss()
    return criterion


def multilevel_ce_loss(criterion, y_pred, y_true, cfg):
    loss = 0
    for level, level_weight in zip(cfg.train.train_on_label_levels, cfg.train.label_levels_weights):
        a = y_pred[str(level)]
        b = y_true[:,level].long()
        level_weight = torch.tensor(level_weight).to(cfg.train.device)
        loss += level_weight * criterion[level](a, b)
    return loss

def print_metrics(metrics, cfg, print_iou= False):

    metrics_names = ['oA','mIoU','mP','mR','mF1','mcAcc']
    metrics_df = pd.DataFrame(columns=metrics_names)
    
    for level in cfg.train.train_on_label_levels:
        
        metric_means = [metrics[level]['overall']['overall_accuracy']]

        if print_iou:
            print(f"IoU at Level: {level}")        
            for cls_name, iou in zip(cfg.label_schema[level].keys(),metrics[level]['classwise']['iou']):
                print(f"\t{cls_name:16}: {iou:.3f}")

        for metric in ['iou','precision','recall','f1_score','class_acc']:
            metric_means.append(np.nanmean(metrics[level]['classwise'][metric]))    
            # print(f"\t{metric:16}: {np.nanmean(metrics[level]['classwise'][metric])}")
        metrics_df.loc[level] = metric_means
    
    print(metrics_df)
    
def get_multilevel_metrics(epoch_predictions, epoch_labels, cfg):


    metrics = {level:dict() for level in cfg.train.train_on_label_levels}
    for level in cfg.train.train_on_label_levels:
        num_classes_at_level = len(cfg.label_schema[level])

        metrics[level]['overall']=dict()
        metrics[level]['classwise']=dict()

        y_pred = epoch_predictions[level]
        y_true = epoch_labels[level]
        k_at_level = torch.arange(num_classes_at_level)[None,:].to(cfg.train.device)
        true_equals_k = k_at_level == y_true[:,None]
        pred_equals_k = k_at_level == y_pred[:,None]

        # initialize tensors to store the metrics
        iou = torch.zeros((num_classes_at_level)).to(cfg.train.device)
        precision = torch.zeros((num_classes_at_level)).to(cfg.train.device)
        recall = torch.zeros((num_classes_at_level)).to(cfg.train.device)
        f1_score = torch.zeros((num_classes_at_level)).to(cfg.train.device)
        class_acc = torch.zeros((num_classes_at_level)).to(cfg.train.device)
        
        # calculate class-wise metrics
        for k in range(num_classes_at_level):
            tp = true_equals_k[:,k] & pred_equals_k[:,k]
            fn = true_equals_k[:,k] & ~pred_equals_k[:,k]
            fp = ~true_equals_k[:,k] & pred_equals_k[:,k]

            iou[k] = tp.sum() / (tp.sum() + fn.sum() + fp.sum())
            precision[k] = tp.sum() / (tp.sum() + fp.sum())
            recall[k] = tp.sum() / (tp.sum() + fn.sum())
            f1_score[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
            class_acc[k] = tp.sum() / true_equals_k[:,k].sum()

        # calculate overall metrics
        overall_accuracy = torch.eq(y_pred, y_true).float().mean().item()
        confusion_matrix = torch.zeros((num_classes_at_level, num_classes_at_level)).to(device=cfg.train.device)
        
        # calculate confusion matrix
        true_labels = torch.argmax(true_equals_k.short(),dim=1)
        pred_labels = torch.argmax(pred_equals_k.short(),dim=1)
        values = torch.ones(len(true_labels)).to(device=cfg.train.device)
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