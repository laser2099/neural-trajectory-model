import argparse
import torch
from torch import nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/src')
from src.train_neural_trajs import *


parser = argparse.ArgumentParser()
parser.add_argument("--save-to", default="/home/dell/zihan/multi_nt/new_src/building_forest.model8.pth")
parser.add_argument("--data", default="/home/dell/zihan/multi_nt/new_src/train_trajs_8agents.npy")
parser.add_argument("--eval-data", default="/home/dell/zihan/multi_nt/new_src/val_trajs_8agents.npy", type=str)
parser.add_argument("--mesh-path", default="/home/dell/zihan/potential_mesh/norm.ply", type=str)
parser.add_argument("--sdf-path", default="/home/dell/zihan/open_source_code/forest_eight/src/building_forest_sdf.pth", type=str)
parser.add_argument("--hidden-dim", default=128, type=int)

parser.add_argument("--A-star-grid-size", default=200, type=int)
parser.add_argument("--A-star-num-traj", default=100, type=int)
parser.add_argument("--A-star-num-agents", default=8, type=int)
parser.add_argument("--A-star-num-step", default=0.5, type=float)

parser.add_argument("--optimizer-lr", default=0.001, type=float)

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


parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--eval-num-workers", default=4, type=int)
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

if args.train_compute_sdf_loss:

    from src.model import finalSDF    
    # sdf_net = SimSDF(128).to(device)
    sdf_net = finalSDF(args.hidden_dim)
    sdf_net = load_model_or_checkoint(args.sdf_path, sdf_net).to(device)
else:
    sdf_net=None

root_path=os.path.dirname(os.path.abspath(__file__))
from utils.utils import check_data
training_data=check_data(args.data,root_path,args,device,sdf_net)
eval_data=check_data(args.eval_data,root_path,args,device,sdf_net)


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
training_dataset = TrajDataset(training_data)

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
    eval_data, 
    batch_eval_func = valid_critierion,
    num_workers=args.eval_num_workers, 
    batch_size=args.eval_batch_size
)

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