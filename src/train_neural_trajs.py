import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from trajs_dataset import TrajDataset
from model_utils import get_model_device
from train_utils import save_checkpoint, load_model_or_checkoint
from eval_multi_trajs import ValidFunction, detach_metrics
from neural_trajectory import TrajTransformer
from neural_trajs_criterions import NeuralTrajsCriterion

import argparse
import numpy as np

def train_trajs_model(
    model, train_dataset, 
    save_to,
    criterion,
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
    for epoch in range(1, max_epoch):
        epoch_loss = []
        loop = tqdm(train_loader)
        for i, data in enumerate(loop):
            model.train()
            points, sdf_vs = data
            points = points.float().to(device)
            sdf_vs = sdf_vs.float().to(device)
            optimizer.zero_grad()
            output = model(points)
            loss, loss_extras = criterion(output, sdf_vs)
            epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            num_updates += 1

            loop.set_description("Epoch [{}/{}] - num_updates[{}]".format(epoch, max_epoch, num_updates))
            loop.set_postfix(loss=loss.item(), valid_loss=valid_loss)

            if num_updates % eval_update_intervals == 0 and valid_func is not None:
                print("\n computing evaluation metrics...")
                valid_loss, valid_extra = valid_func(model)
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    best_valid_metrics = valid_extra
                    print("best valid: ", best_valid_metrics)
                    if num_updates % save_updates_interval == 0: 
                        # torch.save(model, save_to)
                        # checkpoint = save_to + "_{}_".format(num_updates) + ".checkpoint"
                        checkpoint = save_to + ".checkpoint"
                        save_checkpoint(checkpoint, model, epoch, loss, optimizer)
                        print('\nnum_updates: {}, saved checkpoint to: {}'.format(num_updates, checkpoint))
                _, loss_extras = detach_metrics(loss, loss_extras)
                print('\nepoch:{}, loss extras: {}'.format(epoch, str(loss_extras)))
                # print('\nepoch:{}, valid extras: {}'.format(epoch, str(valid_extra)))

        scheduler.step()
        epoch_loss = np.array(epoch_loss).mean()
        best_loss = min(best_loss, epoch_loss)
        print('\nepoch: {},  epoch loss: {}, best loss: {}\n'.format(
            epoch, epoch_loss, best_loss,
            )
        )
        if valid_func is not None:
            valid_loss, valid_extra = valid_func(model)
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                best_valid_metrics = valid_extra
                print('\nepoch:{}, valid metrics: {}; best valid metrics: '.format(epoch, str(valid_extra), str(best_valid_metrics)))
            print('\nepoch:{}, valid metrics: {}'.format(epoch, str(valid_extra)))
    # TODO: add best loss/valid test
    torch.save(model.state_dict(), save_to)
    print('saved model to: ', save_to)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__=='__main__':
    ###
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-to", default="/home/dell/zihan/multi_nt/new_src/building_forest.model8.pth")
    parser.add_argument("--data", default="/home/dell/zihan/multi_nt/new_src/train_trajs_8agents.npy")
    parser.add_argument("--eval-data", default="/home/dell/zihan/multi_nt/new_src/val_trajs_8agents.npy", type=str)
    parser.add_argument("--mesh-path", default="/home/dell/zihan/potential_mesh/norm.ply", type=str)
    parser.add_argument("--sdf-path", default="/home/dell/zihan/multi_nt/new_src/building_forest_sdf.pth.checkpoint", type=str)
    parser.add_argument("--hidden-dim", default=128, type=int)

    parser.add_argument("--safe-dis-sep", default=0.1, type=float)
    parser.add_argument("--safe-time-sep", default=0.05, type=float)
    parser.add_argument("--sdf-thresh", default=0.015, type=float)

    parser.add_argument("--train-compute-travel-time", default=False, type=str2bool)
    parser.add_argument("--train-compute-inter-collision", default=False, type=str2bool)
    parser.add_argument("--train-compute-sdf-loss", default=False, type=str2bool)
    parser.add_argument("--train-compute-distance-loss", default=False, type=str2bool)

    parser.add_argument("--valid-compute-travel-time", default=True, type=str2bool)
    parser.add_argument("--valid-compute-inter-collision", default=True, type=str2bool)
    parser.add_argument("--valid-compute-sdf-loss", default=True, type=str2bool)
    parser.add_argument("--valid-compute-distance-loss", default=True, type=str2bool)


    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--eval-num-workers", default=8, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--eval-batch-size", default=10, type=int)
    parser.add_argument("--max-epoch", default=150, type=int)
    parser.add_argument("--eval-update-intervals", default=800, type=int)
    parser.add_argument("--save-updates-interval", default=1000, type=int)
    args = parser.parse_args()
    print('config: ', args)
    ###
    print(f"PyTorch version: {torch.__version__}")
    # Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
    print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
    print(f"Is MPS available? {torch.backends.mps.is_available()}")

    # Set the device

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():      
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    ##
    from model import finalSDF    
    # sdf_net = SimSDF(128).to(device)
    sdf_net = finalSDF(args.hidden_dim)
    sdf_net = load_model_or_checkoint(args.sdf_path, sdf_net).to(device)
    ###

    ###
    trajs_model = TrajTransformer(args.hidden_dim).to(device)

    train_criterion = NeuralTrajsCriterion(
        matching_criterion=nn.L1Loss(),
        mesh_path=args.mesh_path, 
        sdf_model = sdf_net,
        safe_dis_sep = args.safe_dis_sep,
        safe_time_sep = args.safe_time_sep,
        sdf_thresh = args.sdf_thresh,
        matching_weight = 0.7,
        compute_travel_time=args.train_compute_travel_time,
        compute_inter_collision=args.train_compute_inter_collision,
        compute_sdf_loss=args.train_compute_sdf_loss,
        compute_distance_loss=args.train_compute_distance_loss,          
        trajs_loss_weights = [0.4, 0.4, 0.1, 0.1], # sdf_loss, inter_loss, travel_distance_loss, travel_time_loss
    )
    training_dataset = TrajDataset(args.data)
    ####

    ###
    valid_critierion = NeuralTrajsCriterion(
        matching_criterion=nn.L1Loss(),
        mesh_path=args.mesh_path, 
        sdf_model = sdf_net,
        safe_dis_sep = args.safe_dis_sep,
        safe_time_sep = args.safe_time_sep,
        sdf_thresh = args.sdf_thresh,
        compute_travel_time=args.valid_compute_travel_time,
        compute_inter_collision=args.valid_compute_inter_collision,
        compute_sdf_loss=args.valid_compute_sdf_loss,
        compute_distance_loss=args.valid_compute_distance_loss,   
        matching_weight = 0.7,
        trajs_loss_weights = [0.4, 0.4, 0.1, 0.1], # sdf_loss, inter_loss, travel_distance_loss, travel_time_loss
    )
    


    valid_func = ValidFunction(
        args.eval_data, 
        batch_eval_func = valid_critierion,
        num_workers=args.eval_num_workers, 
        batch_size=args.eval_batch_size
    )
    ###
    train_trajs_model(
        trajs_model, 
        training_dataset, 
        criterion=train_criterion,
        save_to=args.save_to,
        batch_size = args.batch_size, 
        num_workers=args.num_workers, 
        max_epoch=args.max_epoch, 
        eval_update_intervals=args.eval_update_intervals,
        save_updates_interval=args.save_updates_interval,
        valid_func=valid_func)
