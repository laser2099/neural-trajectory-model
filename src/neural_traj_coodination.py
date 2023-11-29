import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader
from coo_dataset import coo_dataset as multi_nt_dataset
from forest_eight.src.src_utils import *
from neural_trajectory import TrajTransformer
from eval_utils import *
    
def cal_distance(point1,point2):
    distance=torch.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(point1[2]-point2[2])**2)
    return distance

def fast_cal_dis(raw_traj):
        a=raw_traj[...,:-1,:]
        b=raw_traj[...,1:,:]
        dis=torch.sum(torch.sum(torch.norm((a-b),dim=-1),dim=-1))
        return dis

def get_gt_sdf(scene,point):
    query_point=o3d.core.Tensor(point.cpu().detach().numpy(),dtype=o3d.core.Dtype.Float32) 
    sdf=scene.compute_signed_distance(query_point)
    return sdf


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_path", default="/home/dell/zihan/potential_mesh/norm.ply")
    parser.add_argument("--safe_dis", default=0.1, type=float)
    parser.add_argument("--safe_time_sep", default=0.05, type=float)
    parser.add_argument("--dim", default=256, type=int)
    parser.add_argument("--num_points", default=128, type=int)
    parser.add_argument("--num_agents", default=8, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--test_data", default='/home/dell/zihan/multi_nt/training_trajs/coordinate_data_final', type=str)
    parser.add_argument("--model_path", default='/home/dell/zihan/open_source_code/forest_eight/building_forest8.pth', type=str)

    args = parser.parse_args()

    safe_dis=args.safe_dis
    num_points=args.num_points
    num_agents=args.num_agents
    
    device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    mesh=o3d.io.read_triangle_mesh(args.mesh_path)
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
    for k,data in tqdm(enumerate(test_loader)):
        test_start_input=data["start"].float().to(device)
        test_end_input=data["end"].float().to(device)
        test_line=data["coo_traj"].float().to(device)
        file_name=data["file_name"]

        with torch.no_grad():
            import time
            t1=time.time()
            test_out = network(test_line)
            t2=time.time()

            cal_time.append(t2-t1)
            gt_traj=test_out[...,1:4].float().to(device)
            norm_test_out=test_out.clone()

  
            traj_with_t=test_out.clone()
            traj_with_t[...,0]=traj_with_t[...,0].cumsum(dim=-1)
            traj_end=gt_traj[:,:,-1,:]
            test_sdf=get_gt_sdf(scene,norm_test_out[...,1:])

            times=np.sum(np.sum(np.sum(test_sdf.numpy()<0,axis=-1),axis=-1)!=0)
            
            env_collision_num.append(times/gt_traj.shape[0])
            
            inter_collision_times=mean_inter_collision_rate(
                                traj_with_t,
                                safe_dis = safe_dis,
                                safe_time_sep=args.safe_time_sep).cpu().numpy()[0]
            
            batch_dis=fast_cal_dis(gt_traj)/num_agents

            traj_distance.append(batch_dis)
            inter_collision_num.append(inter_collision_times)
           
     
    calculation_time=sum(cal_time)/len(cal_time)
    traval_distance=sum(traj_distance)/len(traj_distance)
    collision_rate=sum(env_collision_num)/len(env_collision_num)
    inter_collision=sum(inter_collision_num)/len(inter_collision_num)

    print('inter_collision_num: {}, env_collision_rate: {}, traval_distance: {},cal_time:{}'.format(inter_collision,collision_rate,traval_distance,calculation_time ))


        