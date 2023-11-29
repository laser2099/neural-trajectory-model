import torch
import numpy as np
import heapq 
import os
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import *
from tqdm import tqdm
from utils.interpolation import interpolation
import argparse

class A_star():
    def __init__(self,start,end,sdf_scene,save_path,x_range,y_range,z_range,grid_size,waypoints_num=128) -> None:
        self.start_state=start
        self.end_state=end
        self.save_path=save_path
        self.scene=sdf_scene
        self.waypoints_num=waypoints_num
        self.x_min=x_range[0]
        self.x_max=x_range[1]
        self.y_min=y_range[0]
        self.y_max=y_range[1]
        self.z_min=z_range[0]
        self.z_max=z_range[1]
        self.grid_size=grid_size


    def a_star_init(self):
        side = self.grid_size #PARAM grid size

        
        x_linspace = torch.linspace(-1.8,0.9, side)
        y_linspace = torch.linspace(-1.1,1.1, side)
        z_linspace = torch.linspace(0,0.6, side)

        coods = torch.stack( torch.meshgrid( x_linspace, y_linspace, z_linspace ), dim=-1)
    

        kernel_size = 5 # 100/5 = 20. scene size of 2 gives a box size of 2/20 = 0.1 = drone size
        query_point=o3d.core.Tensor(coods.numpy(),dtype=o3d.core.Dtype.Float32) #输入的点一定要是二维的
        sdf_out=self.scene.compute_signed_distance(query_point) #coods.shape=(100,100,100,3) -> (1e6,3) output.shape=(100,100,100)
        output=torch.tensor(sdf_out.numpy())
        maxpool = torch.nn.MaxPool3d(kernel_size = kernel_size)
        #PARAM cut off such that neural network outputs zero (pre shifted sigmoid)

        # 20, 20, 20
        occupied = -maxpool(-output[None,None,...])[0,0,...] < -0.1
        
        grid_size = side//kernel_size

        #convert to index cooredinates
        start_grid_float = grid_size*(self.start_state[:3] + 1)/2
        end_grid_float   = grid_size*(self.end_state  [:3] + 1)/2
        start = tuple(int(start_grid_float[i]) for i in range(3) )
        end =   tuple(int(end_grid_float[i]  ) for i in range(3) )

        #print(start, end)
        try:
            path = self.astar(occupied, start, end)

            # convert from index cooredinates
            squares =  2* (torch.tensor( path, dtype=torch.float)/grid_size) -1

            #adding way
            states = torch.cat( [squares, torch.zeros( (squares.shape[0], 1) ) ], dim=-1)

            #prevents weird zero derivative issues
            randomness = torch.normal(mean= 0, std=0.001*torch.ones(states.shape) )
            states += randomness

            # smooth path (diagram of which states are averaged)
            # 1 2 3 4 5 6 7
            # 1 1 2 3 4 5 6
            # 2 3 4 5 6 7 7
            prev_smooth = torch.cat([states[0,None, :], states[:-1,:]],        dim=0)
            next_smooth = torch.cat([states[1:,:],      states[-1,None, :], ], dim=0)
            states = (prev_smooth + next_smooth + states)/3

            self.states = states[...,:3].clone().detach()
            gt_traj=self.get_gt_traj(self.states,self.waypoints_num)
            query_point=o3d.core.Tensor(gt_traj,dtype=o3d.core.Dtype.Float32) 
            sdf=self.scene.compute_signed_distance(query_point)
            if all(sdf>0):
                np.savetxt(self.save_path,gt_traj,delimiter=',')
                print('--------------------------')
                print('Successfully generating one trajectory!')
                print('--------------------------')
        except Exception:
            return 
    
    def get_gt_traj(self,data,num_points):
        final_data=np.zeros([num_points,3])
        inter=interpolation(num_points)
        x=data[...,0].tolist()
        y=data[...,1].tolist()
        z=data[...,2].tolist()
        x_new,y_new,z_new=inter.cubicSplineInterpolate(x,y,z)
        final_x=x_new[0:-1:len(x)-1]
        final_y=y_new[0:-1:len(y)-1]
        final_z=z_new[0:-1:len(z)-1]
        final_data[...,0]=np.array(final_x)
        final_data[...,1]=np.array(final_y)
        final_data[...,2]=np.array(final_z)
        return final_data

    def astar(self,occupied, start, goal):
        def heuristic(a, b):
            return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)

        def inbounds(point):
            for x, size in zip(point, occupied.shape):
                if x < 0 or x >= size: return False
            return True

        neighbors = [( 1,0,0),(-1, 0, 0),
                    ( 0,1,0),( 0,-1, 0),
                    ( 0,0,1),( 0, 0,-1)]

        close_set = set()

        came_from = {}
        gscore = {start: 0}

        assert not occupied[start]
        assert not occupied[goal]

        open_heap = []
        heapq.heappush(open_heap, (heuristic(start, goal), start))

        while open_heap:
            current = heapq.heappop(open_heap)[1]

            if current == goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                assert current == start
                data.append(current)
                return list(reversed(data))

            close_set.add(current)

            for i, j, k in neighbors:
                neighbor = (current[0] + i, current[1] + j, current[2] + k)
                if not inbounds( neighbor ):
                    continue

                if occupied[neighbor]:
                    continue

                tentative_g_score = gscore[current] + 1

                if tentative_g_score < gscore.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score

                    fscore = tentative_g_score + heuristic(neighbor, goal)
                    node = (fscore, neighbor)
                    if node not in open_heap:
                        heapq.heappush(open_heap, node) 

        raise ValueError("Failed to find path!")
    
def start_generate_A_star(mesh_path,root_path,grid_size,step,num_agents,num_traj):
    x_range,y_range,z_range=get_mesh_range(mesh_path)
    print('--------------------------')
    print('Get the range of the mesh!')
    print('--------------------------')

    mesh=o3d.io.read_triangle_mesh(mesh_path)
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene=o3d.t.geometry.RaycastingScene()
    _=scene.add_triangles(mesh)
    num=0
    sta_end_dir=root_path+'_sta_end_pairs'
    save_path=root_path+'_A_star_trajs'
    if os.path.exists(save_path):
        pass
    else:
        os.mkdir(save_path)
    if os.path.exists(sta_end_dir):
        sta_end_lst=os.listdir(sta_end_dir)
        if len(sta_end_lst)==0:
            from utils.generate_single_data import generate_start_end,generate_sta_end_pair
            csv_path=generate_start_end(mesh_path,step,x_range,y_range,z_range,root_path+'_')
            generate_sta_end_pair(csv_path,sta_end_dir+'/',num_agents,num_traj)
            sta_end_lst=os.listdir(sta_end_dir)
    else:
        os.makedirs(sta_end_dir)
        from utils.generate_single_data import generate_start_end,generate_sta_end_pair
        csv_path=generate_start_end(mesh_path,step,x_range,y_range,z_range,root_path+'/')
        generate_sta_end_pair(csv_path,sta_end_dir,num_agents,num_traj)
        sta_end_lst=os.listdir(sta_end_dir)
    
    for csv in tqdm(sta_end_lst):
        csv_name=sta_end_dir+'/'+csv
        data=np.genfromtxt(csv_name,delimiter=',')
        for i in range(len(data)):
            sta_end_pair=data[i]
            start=sta_end_pair[:3]
            end=sta_end_pair[3:]
            traj_path=save_path+'/'+str(num)+'.csv'
            a_star_generator=A_star(start,end,scene,traj_path,x_range,y_range,z_range,grid_size)
            a_star_generator.a_star_init()
            num+=1

def get_mesh_range(mesh_path):
    import open3d as o3d
    mesh=o3d.io.read_triangle_mesh(mesh_path)
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    x_lst=[]
    y_lst=[]
    z_lst=[]
    for ver in mesh.vertex.positions:
        x=ver[0]
        y=ver[1]
        z=ver[2]
        x_lst.append(x)
        y_lst.append(y)
        z_lst.append(z)

    x_min=min(x_lst).item()
    y_min=min(y_lst).item()
    z_min=min(z_lst).item()
    x_max=max(x_lst).item()
    y_max=max(y_lst).item()
    z_max=max(z_lst).item()
    max_val=max([x_min,y_min,z_min,x_max,y_max,z_max])
    min_val=min([x_min,y_min,z_min,x_max,y_max,z_max])
    if max_val>2 or min_val<-2:
        raise ValueError("Mesh range error! You need to use blender to reduce the scope of the mesh!")
    return [x_min,x_max],[y_min,y_max],[z_min,z_max]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-to", default="/home/dell/zihan/multi_nt/A_star_trajs/", type=str)
    parser.add_argument("--mesh-path", default="/home/dell/zihan/potential_mesh/norm.ply", type=str)
    parser.add_argument("--start-end-path", default="/home/dell/zihan/multi_nt/sta_end_pairs", type=str)
    parser.add_argument("--x-range", default=[-1.8,0.9], type=list)
    parser.add_argument("--y-range", default=[-1.1,1.1], type=list)
    parser.add_argument("--z-range", default=[0,0.6], type=list) #the coordinates range of the mesh map
    parser.add_argument("--grid-size", default=200, type=int)

    args = parser.parse_args()
    sta_end_dir=args.start_end_path
    sta_end_lst=os.listdir(sta_end_dir)
    save_path=args.save_to
    x_range=args.x_range
    y_range=args.y_range
    z_range=args.z_range
    grid_size=args.grid_size
    mesh_path=args.mesh_path

    mesh=o3d.io.read_triangle_mesh(mesh_path)
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene=o3d.t.geometry.RaycastingScene()
    _=scene.add_triangles(mesh)
    num=0
    for csv in tqdm(sta_end_lst):
        csv_name=sta_end_dir+'/'+csv
        data=np.genfromtxt(csv_name,delimiter=',')
        for i in range(len(data)):
            sta_end_pair=data[i]
            start=sta_end_pair[:3]
            end=sta_end_pair[3:]
            traj_path=save_path+str(num)+'.csv'
            a_star_generator=A_star(start,end,scene,traj_path,x_range,y_range,z_range,grid_size)
            a_star_generator.a_star_init()
            num+=1


     

        