import numpy as np
import os 
import shutil
import open3d as o3d
from tqdm import tqdm
from eval_utils import *

def check_env_collision(data,scene):
    sdf=get_gt_sdf(scene,data)

    return sum([any(sdf_ele<=0) for sdf_ele in sdf])==0


def get_gt_sdf(scene,point):
    query_point=o3d.core.Tensor(point,dtype=o3d.core.Dtype.Float32) #输入的点一定要是二维的
    sdf=scene.compute_signed_distance(query_point)
    return sdf

def check_inter_collision(num_agents,traj,safe_dis,num_points=128):
        time_lst=np.zeros([num_agents,num_agents])
        
    
        collision_lst=[]
        for i in range(num_points):
            ele=traj[:,i,:]
            time=ele[:,0]
            for j in range(len(time)):
                time_dis=time[j]-time
                time_lst[j]=abs(time_dis)
            indices=np.nonzero(time_lst<0.05)
            for indice in indices:
                point1=indice[0]
                point2=indice[1]
                if point1 != point2:
                    pos1=ele[point1,1:]
                    pos2=ele[point2,1:]
                    dis=cal_distance(pos1,pos2)
                    if dis<safe_dis:
                        collision_lst.append(1)
        collision=sum(collision_lst)         
        return collision==0

def cal_distance(point1,point2):
        distance=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(point1[2]-point2[2])**2)
        return distance

def start_check(data_dir,train_dir,safe_dis,mesh_path):
    mesh=o3d.io.read_triangle_mesh(mesh_path)
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene=o3d.t.geometry.RaycastingScene()
    _=scene.add_triangles(mesh)
    data_lst=os.listdir(data_dir)
    for formation_data in tqdm(data_lst):
        formation_file=data_dir+formation_data
        csv_lst=os.listdir(formation_file)
        if len(csv_lst)==0:
            continue
        num_agents=len(csv_lst)
        final_data=np.zeros((num_agents,128,4))
        for j,csv_file in enumerate(csv_lst):
            csv_name=data_dir+formation_data+'/'+csv_file
            data=np.genfromtxt(csv_name,delimiter=',')
            final_data[j,...]=data
        env_flag=check_env_collision(final_data[...,1:],scene)
        inter_flag=mean_inter_collision_rate(
        torch.FloatTensor(final_data).unsqueeze(0),
        safe_dis = 0.1,
        safe_time_sep=0.05)
        # inter_flag=check_inter_collision(num_agents,final_data,safe_dis)
        


    pass
if __name__=="__main__":
    data_dir='/home/dell/zihan/multi_nt/training_trajs/coo_data/'
    train_dir='/home/dell/zihan/multi_nt/training_trajs/coordi_data/'
    safe_distance=0.1
    #mesh_path='/home/dell/zihan/potential_mesh/City.ply'
    mesh_path='/home/dell/zihan/potential_mesh/norm.ply'
    start_check(data_dir,train_dir,safe_distance,mesh_path)
