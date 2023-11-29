import open3d as o3d
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
import argparse

def generate_start_end(mesh_path,step,x_range,y_range,z_range,store_path):
    mesh=o3d.io.read_triangle_mesh(mesh_path)
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene=o3d.t.geometry.RaycastingScene()
    _=scene.add_triangles(mesh)
    pos_lst=[]
    for x in tqdm(np.arange(x_range[0],x_range[1],step)):
        for y in np.arange(y_range[0],y_range[1],step):
            for z in np.arange(z_range[0],z_range[1],step):
                pos=[x,y,z]
                query_point=o3d.core.Tensor([pos],dtype=o3d.core.Dtype.Float32) #输入的点一定要是二维的
                sdf=scene.compute_signed_distance(query_point)
                if sdf>0.05:
                    pos_lst.append(pos)
    sta_end = np.asarray(pos_lst)
    csv_path=store_path+'sta_end.csv'
    np.savetxt(csv_path, sta_end, delimiter=",")
    return csv_path


def generate_sta_end_pair(csv_path,pair_dir,num_agents,num_traj):
    lst=np.genfromtxt(csv_path,delimiter=',').tolist()
    num_lst=[]
    for i in range(len(lst)):
        num_lst.append(i)

    for m in tqdm(range(1,num_traj)):
        csv_file=pair_dir+str(m)+'.csv'
        pos_lst=[]
        pop_lst=[]
        for n in range(num_agents):
            choose=np.random.choice(num_lst,2,replace=False)#从一个0-的array任意选出两个值
            pop_lst.append(choose[0])#删除列表中加入这两个值
            pop_lst.append(choose[1])
            num_lst.remove(choose[0])
            num_lst.remove(choose[1])
            pos=lst[choose[0]].copy()
            pos.extend(lst[choose[1]])#选出这两个值
            pos_lst.append(pos)
        with open(csv_file,"w",newline="") as file:
            writer=csv.writer(file)
            for row in pos_lst:
                writer.writerow(row)
        num_lst.extend(pop_lst)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-to", default="/home/dell/zihan/multi_nt/", type=str)
    parser.add_argument("--mesh-path", default="/home/dell/zihan/potential_mesh/norm.ply", type=str)
    parser.add_argument("--start-end-path", default="/home/dell/zihan/multi_nt/sta_end/", type=str)
    parser.add_argument("--x-range", default=[-1.8,0.9], type=list)
    parser.add_argument("--y-range", default=[-1.1,1.1], type=list)
    parser.add_argument("--z-range", default=[0,0.6], type=list) #the coordinates range of the mesh map
    parser.add_argument("--step", default=0.01, type=float)
    parser.add_argument("--num-traj",default=101,type=int)
    parser.add_argument("--num-agents",default=8,type=int)
    args = parser.parse_args()

    mesh_path=args.mesh_path
    step=args.step
    x_range=args.x_range
    y_range=args.y_range
    z_range=args.z_range
    store_path=args.save_to
    pair_dir=args.start_end_path
    num_agents=args.num_agents
    num_traj=args.num_traj

    csv_path=generate_start_end(mesh_path,step,x_range,y_range,z_range,store_path)
    generate_sta_end_pair(csv_path,pair_dir,num_agents,num_traj)