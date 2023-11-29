import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader
from raw_nt_dataset import nt_dataset as multi_nt_dataset
from forest_eight.src.src_utils import *
from neural_trajectory import TrajTransformer
from eval_utils import *
import time

def fast_cal_dis(raw_traj):
        a=raw_traj[...,:-1,:]
        b=raw_traj[...,1:,:]
        dis=torch.sum(torch.sum(torch.norm((a-b),dim=-1),dim=-1))
        return dis

def get_gt_sdf(scene,point):
    query_point=o3d.core.Tensor(point.cpu().detach().numpy(),dtype=o3d.core.Dtype.Float32) #输入的点一定要是二维的
    sdf=scene.compute_signed_distance(query_point)
    return sdf


def cal_stop_time(num_points,traj,end_pos):
    stop=[]
    for m in range(len(traj)):
        sub_traj=traj[m]
        stop_time=[]
        for n in range(len(sub_traj)):
            sing_traj=sub_traj[n]
            end=end_pos[m][n]
            for i in range(num_points-1):
                if sing_traj[i][1]==end[1] and sing_traj[i][2]==end[2] and sing_traj[i][3]==end[3]:
                    continue
                x_flag=sing_traj[i][1]-sing_traj[i+1][1]
                y_flag=sing_traj[i][2]-sing_traj[i+1][2]
                z_flag=sing_traj[i][3]-sing_traj[i+1][3]
                if abs(x_flag)<0.0001 and abs(y_flag) <0.0001 and abs(z_flag)<0.0001:
                    stop_time.append(sing_traj[i+1][0]-sing_traj[i][0])
        stop.append(sum(stop_time)/8)

    return sum(stop)/len(stop)

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", default='/home/dell/zihan/open_source_code/stonehenge_eight/stone.ply')
    parser.add_argument("--safe_dis", default=0.05, type=float)
    parser.add_argument("--safe_time_sep", default=0.05, type=float)
    parser.add_argument("--dim", default=256, type=int)
    parser.add_argument("--num_points", default=128, type=int)
    parser.add_argument("--num_agents", default=8, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--test_data", default='/home/dell/zihan/multi_nt/training_trajs/eight_test_final', type=str)
    parser.add_argument("--model_path", default='/home/dell/zihan/open_source_code/forest_eight/building_forest8.pth', type=str)

    args = parser.parse_args()

    safe_dis=args.safe_dis
    num_points=args.num_points
    num_agents=args.num_agents
    device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    mesh_path=args.mesh_path
    mesh=o3d.io.read_triangle_mesh(mesh_path)
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene=o3d.t.geometry.RaycastingScene()
    _=scene.add_triangles(mesh)

    test_root=args.test_data
   

    test_dataset=multi_nt_dataset(test_root,num_points)
    test_loader = DataLoader(test_dataset,
                                        shuffle=False,
                                        batch_size=args.batch_size,
                                        num_workers=0,
                                        pin_memory=True,
                                        drop_last=True
                                        )
    env_collision_num=[]
    inter_collision_num=[]
    stop_times=[]
    traj_distance=[]
    cal_time=[]
    test_loss_lst=[]
    criterion=nn.L1Loss()

    network=TrajTransformer(dim=args.dim,num_pp=num_points)
    network.load_state_dict(torch.load(args.model_path))

    network.to(device).eval()
    uncollision_file=[]
    inter_col_file=[]
    env_col_file=[]

    import os
    data_dir='/home/dell/zihan/multi_nt/training_trajs/stonehenge_test_traj'
    data_lst=os.listdir(data_dir)
    coll_file=[]
    for k,file_name in tqdm(enumerate(data_lst)):
        flag=0
        file=data_dir+'/'+file_name
        trajectory=torch.zeros([8,128,4])
        csv_lst=os.listdir(file)
        for id,csv in enumerate(csv_lst):
            csv_name=file+'/'+csv
            data=torch.tensor(np.genfromtxt(csv_name,delimiter=','))
            trajectory[id]=data

        trajectory=trajectory.unsqueeze(0)
        gt_traj=trajectory[...,1:4].float()
        norm_test_out=trajectory.clone()

        traj_with_t=trajectory.clone()


        test_sdf=get_gt_sdf(scene,norm_test_out[...,1:])

        times=np.sum(np.sum(np.sum(test_sdf.numpy()<0,axis=-1),axis=-1)!=0)
        
        env_collision_num.append(times/gt_traj.shape[0])
        
        inter_collision_times=mean_inter_collision_rate(
                            traj_with_t,
                            safe_dis = safe_dis,
                            safe_time_sep=args.safe_time_sep).numpy()[0]
        if inter_collision_times==0 and times==0:
            coll_file.append(file_name)

        batch_dis=fast_cal_dis(gt_traj)/num_agents
   
        traj_distance.append(batch_dis)
        inter_collision_num.append(inter_collision_times!=0)
 

    traval_distance=sum(traj_distance)/len(traj_distance)
    collision_rate=sum(env_collision_num)/len(env_collision_num)
    inter_collision=sum(inter_collision_num)/len(inter_collision_num)
    print('inter_collision_num: {}, env_collision_rate: {}, traval_distance: {}'.format(inter_collision,collision_rate,traval_distance))

   
    