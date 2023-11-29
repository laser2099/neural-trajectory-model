import torch

from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

from trajs_dataset import TrajDataset
from model_utils import get_model_device

def aggregate_eval_func(metrics, loss, loss_extras, batch):
    num_samples = metrics.get('num_samples', 0)
    best_loss = metrics.get('best_loss', 1e8)
    metrics['num_samples'] = num_samples + batch.size(0)
    bsz = batch.size(0)
    metrics['acc_loss'] = (
        loss * batch.size(0) / metrics['num_samples'] 
        + metrics.get('acc_loss', 0) * num_samples / metrics['num_samples']
    )
    # TODO: metrics['extras'] = aggreate_extras(loss_extras)
    if "acc_extras" not in metrics:
        metrics['acc_extras'] = {}
    acc_extras = metrics["acc_extras"]
    for k, v in loss_extras.items():
        acc_v = acc_extras.get(k, 0.0)
        v = v or 0
        acc_v = v * bsz / metrics['num_samples'] + acc_v * num_samples / metrics['num_samples']
        acc_extras[k] = acc_v

    # metrics['last_loss_extras'] = loss_extras
    # if best_loss > loss:
    #     metrics['best_loss'] = loss
    #     metrics['best_loss_extras'] = loss_extras
    return metrics

def detach_metrics(loss, metrics):
    loss = loss.item()
    metrics = {
        k: v.item() if v is not None else None
        for k, v in metrics.items()
    }
    return loss, metrics


def eval_ntrajs_model(
    model, dataset, 
    batch_eval_func,
    aggregate_eval_func,
    batch_size = 1000,
    num_workers=8,
):
    if len(dataset) < 0:
        print("eval dataset is empty")
        return None
    device = get_model_device(model)
    test_loader = DataLoader(dataset,
                        shuffle=False,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=True,
                        drop_last=False,
                        )
    model.eval()
    eval_metrics = {}
    for i, data in tqdm(enumerate(test_loader)):
        lines, gt_trajs = data
        lines = lines.float().to(device)
        gt_trajs = gt_trajs.float().to(device)    
        with torch.no_grad():
            output = model(lines)
            valid_loss, valid_extras = batch_eval_func(output, gt_trajs)
            valid_loss, valid_extras = detach_metrics(valid_loss, valid_extras)
            eval_metrics = aggregate_eval_func(
                eval_metrics,
                valid_loss,
                valid_extras,
                batch=lines,
            )
    return eval_metrics


class ValidFunction:
    def __init__(
        self,
        eval_data_path,
        batch_eval_func,
        aggregate_eval_func = aggregate_eval_func,
        batch_size = 1000,
        num_workers=8,
    ):
        self.num_workers = num_workers
        self.batch_eval_func = batch_eval_func
        self.aggregate_eval_func = aggregate_eval_func
        self.batch_size = batch_size
        self.dataset = TrajDataset(eval_data_path)
        print("eval data sizes: ", len(self.dataset))
    
    def __call__(self, model):
        metrics = eval_ntrajs_model(
            model, 
            self.dataset, 
            self.batch_eval_func, 
            self.aggregate_eval_func, 
            self.batch_size, 
            self.num_workers)
        return metrics.get("acc_loss", 1e10), metrics
