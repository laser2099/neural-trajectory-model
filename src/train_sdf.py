import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from env_dataset import SDFDataset
from sdf_models import SimSDF
from eval_sdf import ValidFunction
from model_utils import get_model_device
from train_utils import save_checkpoint, get_better_device, load_checkpoint, last_checkpoint, best_checkpoint
from model import SimpleSDF, finalSDF

import numpy as np
import os


def train_sdf(
    args,
    model, train_dataset, 
    save_to,
    criterion= nn.MSELoss(),
    sdf_scale_up = 100,
    batch_size = 1000,
    max_lr = 1e-3,
    min_lr = 1e-6,
    max_epoch = 3000,
    num_workers = 8,
    valid_func = None,   
    save_epoch_interval = 1,
    save_updates_interval = 1000,
    eval_update_intervals = 200,
):
    # TODO: add recover training from checkpoint using load_checkpoint
    device = get_model_device(model)

    
    optimizer = optim.AdamW(model.parameters(), lr=max_lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=min_lr)
    start_epoch = 1
    if args.recover_from_last_checkpoint and os.path.exists(last_checkpoint(save_to)):
        print('Loading from last checkpoint: ', last_checkpoint(save_to))
        model, _, start_epoch, loss = load_checkpoint(last_checkpoint(save_to), model, optimizer)

    train_loader = DataLoader(train_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=True
                            )
    print('training data size: ', len(train_dataset))
    best_loss = 1e8
    best_valid_loss = 1e8
    best_valid_metrics = {}
    valid_loss = 1e8
    num_updates = 0
    
    for epoch in range(start_epoch, max_epoch):
        epoch_loss = []
        loop = tqdm(train_loader)
        for i, data in enumerate(loop):
            model.train()
            points, sdf_vs = data
            points = points.float().to(device)
            sdf_vs = sdf_vs.float().to(device)
            optimizer.zero_grad()
            output = model(points)
            loss = criterion(output * sdf_scale_up, sdf_vs * sdf_scale_up)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            num_updates += 1

            loop.set_description("Epoch [{}/{}] - num_updates[{}]".format(epoch, max_epoch, num_updates))
            loop.set_postfix(loss=loss.item(), valid_loss=valid_loss)

            if num_updates % eval_update_intervals == 0 and valid_func is not None:
                print("\n computing evaluation metrics...")
                metrics = valid_func(model)
                valid_loss = metrics["overall"]
                if best_valid_loss > metrics["overall"]:
                    best_valid_loss = metrics["overall"]
                    best_valid_metrics = metrics
                    print("best valid: ", best_valid_metrics)
                    save_checkpoint(args, best_checkpoint(save_to), model, epoch, loss, optimizer)
                    print('\nnum_updates: {}, saved best valid model to: {}'.format(num_updates, best_checkpoint(save_to)))

                print('\nepoch:{}, valid metrics: {}'.format(epoch, str(metrics)))
            if num_updates % save_updates_interval == 0: 
                # torch.save(model, save_to)
                # checkpoint = save_to + "_{}_".format(num_updates) + ".checkpoint"
                checkpoint = last_checkpoint(save_to)
                save_checkpoint(args, checkpoint, model, epoch, loss, optimizer)
                print('\nnum_updates: {}, saved model to: {}'.format(num_updates, checkpoint))
        scheduler.step()
        epoch_loss = np.array(epoch_loss).mean()
        best_loss = min(best_loss, epoch_loss)
        print('\nepoch: {},  epoch loss: {}, best loss: {}\n'.format(
            epoch, epoch_loss, best_loss,
            )
        )
        if valid_func is not None:
            metrics = valid_func(model)
            if best_valid_loss > metrics["overall"]:
                best_valid_loss = metrics["overall"]
                best_valid_metrics = metrics
                print('\nepoch:{}, valid metrics: {}; best valid metrics: '.format(epoch, str(metrics), str(best_valid_metrics)))
            print('\nepoch:{}, valid metrics: {}'.format(epoch, str(metrics)))
    # TODO: add best loss/valid test
    # torch.save(model, save_to)
    save_checkpoint(args, save_to + "final.checkpoint", model, epoch, loss, optimizer)    
    print('saved model to: ', save_to)




if __name__=='__main__':
    ###
    import argparse
    parser = argparse.ArgumentParser()
    root = "/home/tang/workspace/zihan-work/open_source_code20231005/aamas"
    eval_data = f"{root}/data/env_data/building_forest_sdf.eval.npy"
    train_data =  f"{root}/data/env_data/building_forest_sdf.train10e7.npy"

    train_data = f"{root}/data/env_data/building_forest_sdf.train10e7.v3.npy"

    parser.add_argument("--save-to", default="/home/tang/workspace/experiments/building_forest.tri_posemb.mlp_out.v5.train10e7.pth")
    parser.add_argument("--data", default=train_data)
    parser.add_argument("--hidden-dim", default=256, type=int)
    parser.add_argument("--eval-data", default=eval_data, type=str)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--eval-num-workers", default=2, type=int)
    parser.add_argument("--batch-size", default=3 * 10**3, type=int)
    parser.add_argument("--eval-batch-size", default=2 * 10**3, type=int)
    parser.add_argument("--max-epoch", default=100, type=int)
    parser.add_argument("--recover-from-last-checkpoint", default=True, type=bool)
    parser.add_argument("--eval-update-intervals", default=1000, type=int)
    parser.add_argument("--save-updates-interval", default=1000, type=int)
    args = parser.parse_args()
    ###
    print(f"PyTorch version: {torch.__version__}")
    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    # Set the device
    device = get_better_device()


    print(f"Using device: {device}")

    sdf_dataset = SDFDataset(args.data)
    # from model import SimpleSDF
    # sdf_net = SimSDF(128).to(device)
    # sdf_net = finalSDF(args.hidden_dim).to(device)
    sdf_net = SimpleSDF(args.hidden_dim, num_layers=6).to(device)

    valid_func = ValidFunction(args.eval_data, num_workers=args.eval_num_workers, batch_size=args.eval_batch_size)

    train_sdf(
        args,
        sdf_net, 
        sdf_dataset, 
        save_to=args.save_to,
        sdf_scale_up = 100,
        batch_size = args.batch_size, 
        num_workers=args.num_workers, 
        max_epoch=args.max_epoch, 
        eval_update_intervals=args.eval_update_intervals,
        save_updates_interval=args.save_updates_interval,
        valid_func=valid_func)