import torch
from torch import optim
from typing import Mapping
from torch import nn

import numpy as np
from model_utils import get_model_device
import os

import sys
import logging

def config_logging():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )




def last_checkpoint(save_to):
    return save_to + ".last.checkpoint"

def best_checkpoint(save_to):
    return save_to + ".best.checkpoint"


def get_better_device(cuda_rank=0):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_rank}')
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"getting device: {device}")
    return device

def save_checkpoint(args, save_to, model, epoch, current_loss, optimizer):
    #TODO
    torch.save({
                'config': args,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                }, save_to)
    

def load_checkpoint(
        from_path, 
        model, 
        lr=0.001, momentum=0.9,
    ):
    device = get_model_device(model)

    checkpoint = torch.load(from_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model,  epoch, loss

def load_model_or_checkpoint(
        from_path, 
        model, 
        lr=0.001, momentum=0.9,        
):
    device = get_model_device(model)

    loaded = torch.load(from_path, map_location=device)
    if isinstance(loaded, Mapping): 
        if 'model_state_dict' in loaded:
            # TODO: recover lr and momentum from checkpoint
            model.load_state_dict(loaded['model_state_dict'])
        else:
            model.load_state_dict(loaded)
        return model
    elif isinstance(loaded, nn.Module):
        model = loaded
        # print(type(loaded), isinstance(loaded, nn.Module))
        return model
    else:
        raise NotImplementedError("not supported format")
    
def load_model_or_checkoint(
        from_path, 
        model, 
        lr=0.001, momentum=0.9,        
):
    loaded = torch.load(from_path)
    if isinstance(loaded, Mapping): 
        if 'model_state_dict' in loaded:
            model, loss, loss = load_checkpoint(from_path, model, lr, momentum)
        else:
            model.load_state_dict(loaded)
        return model
    elif isinstance(loaded, nn.Module):
        model = loaded
        # print(type(loaded), isinstance(loaded, nn.Module))
        return model
    else:
        raise NotImplementedError("not supported format")
