import torch
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import *
from torch.utils.data import DataLoader
from utils.A_start_optimize_dataset import optimize_dataset
import argparse

class traj_optimizer():
    def __init__(self,traj,lr,num_epoch,mesh_path,safe_dis,device,sdf_network,save_dir,sdf_thresh=0.25) -> None:
        self.traj=traj
        self.lr=lr
        self.epochs_init=num_epoch
        self.mesh_path=mesh_path
        self.safe_dis=safe_dis
        self.device=device
        self.sdf_network=sdf_network
        self.sdf_thresh=sdf_thresh
        self.save_dir=save_dir

    def params(self):
        return [self.traj]
    
    def get_inter_collision_loss(self,num_agents,num_points,traj,safe_dis,device):
        time_lst=torch.zeros([num_agents,num_agents]).to(device)
        collisions=[]
        for sub_traj in traj:
            collision_lst=[]
            for i in range(num_points):
                ele=sub_traj[:,i,:]
                time=ele[:,0]
                for j in range(len(time)):
                    time_dis=time[j]-time
                    time_lst[j]=abs(time_dis)
                indices=torch.nonzero(time_lst<0.05)
                for indice in indices:
                    point1=indice[0]
                    point2=indice[1]
                    if point1 != point2:
                        pos1=ele[point1,1:]
                        pos2=ele[point2,1:]
                        dis=self.cal_distance(pos1,pos2).unsqueeze(-1)
                        collision_loss=torch.max(safe_dis-dis,torch.zeros([1]).to(device))
                        collision_lst.append(collision_loss)
            collision=sum(collision_lst)         
            collisions.append(collision)
        return sum(collisions)

    def cal_distance(self,point1,point2):
        distance=torch.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(point1[2]-point2[2])**2)
        return distance
    
    def get_sdf_loss(self,raw_traj):
        for param in self.sdf_network.parameters():
            param.requires_grad = False
            sdf_lst=self.sdf_network(raw_traj.view(-1,3))
        sdf_gt=torch.zeros([sdf_lst.shape[0]]).to(self.device)
        sdf_loss=sum(torch.max((self.sdf_thresh-sdf_lst),sdf_gt))
        return sdf_loss
    
    def load_mesh(self,mesh_path):
        from kaolin.ops.mesh import index_vertices_by_faces
        test_mesh = o3d.io.read_triangle_mesh(mesh_path)
        self.vertices = torch.tensor(np.asarray(test_mesh.vertices)).float().unsqueeze(0).to(self.device)
        self.faces = torch.tensor(np.asarray(test_mesh.triangles),dtype=torch.long).to(self.device)
        self.face_vertices = index_vertices_by_faces(self.vertices, self.faces)[0].to(self.device)

    def kal_mean_sdf_loss(self, trajs):
        import kaolin as kal
        coordinates = trajs[...,1:].view(-1,3)
        self.load_mesh(self.mesh_path)
        distance, _, _= kal.metrics.trianglemesh._unbatched_naive_point_to_mesh_distance(coordinates, self.face_vertices)
        flag = kal.ops.mesh.check_sign(self.vertices, self.faces, coordinates.reshape(1,-1,3)).squeeze(0)
        quan_flag = ~flag*2-1
        zeros = torch.zeros([1], device=trajs.device)
        sdf_lst = quan_flag * torch.sqrt(distance)
        sdf_loss = torch.max((self.sdf_thresh - sdf_lst), zeros).mean()
        return sdf_loss

        
    def get_dis(self,raw_traj):
        bs,num_agents,horizon,dim=raw_traj.shape
        s_traj=raw_traj.view(bs*num_agents,horizon,dim)
        total_dis=[]
        for i in range(horizon-1):
            current_pos=s_traj[:,i,:]
            next_pos=s_traj[:,i+1,:]
            sum_dis=torch.sum(torch.sqrt(torch.sum((current_pos-next_pos)**2,dim=1)))
            total_dis.append(sum_dis)
        return sum(total_dis)
    
    def fast_cal_dis(self,raw_traj):
        a=raw_traj[...,:-1,:]
        b=raw_traj[...,1:,:]
        dis=torch.sum(torch.sum(torch.norm((a-b),dim=-1),dim=-1))
        return dis

    def start_optimize(self,traj_id):
        opt = torch.optim.Adam(self.params(), lr=self.lr, capturable=True)
        data_dir=self.save_dir
        traj_dir=data_dir+str(traj_id)
        raw_length=self.fast_cal_dis(self.traj[...,1:])
        if os.path.exists(traj_dir):
            pass
        else:
            os.makedirs(traj_dir)
        for it in range(self.epochs_init):
            opt.zero_grad()
            self.epoch = it
            loss,env_loss,inter_loss,dis_loss = self.total_cost()
            print(it, dis_loss)
            print('----------------------------')
            print(env_loss,inter_loss)
            #if env_loss==0 and inter_loss==0:
            if inter_loss==0 and dis_loss<1.1*raw_length:
                with torch.no_grad():
                    final_traj=self.params()[0][0].detach().cpu().numpy()
                    if self.start_check(final_traj):
                        for agent_id in range(final_traj.shape[0]):
                            file_name=traj_dir+'/'+str(agent_id)
                            single_traj=final_traj[agent_id]
                            np.savetxt(file_name, single_traj, delimiter=",")
                break 
            else:
                loss.backward()
                opt.step()
                if it > 50:
                    break
            
        return self.traj
    
    def start_check(self,final_data):
        from src.eval_utils import mean_inter_collision_rate

        mesh=o3d.io.read_triangle_mesh(self.mesh_path)
        mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene=o3d.t.geometry.RaycastingScene()
        _=scene.add_triangles(mesh)
        
        env_flag=self.check_env_collision(final_data[...,1:],scene)
        inter_flag=mean_inter_collision_rate(
        torch.FloatTensor(final_data).unsqueeze(0),
        safe_dis = 0.05,
        safe_time_sep=0.05)
        return env_flag and inter_flag<1e-2

    def check_env_collision(self,data,scene):
        sdf=self.get_gt_sdf(scene,data)

        return sum([any(sdf_ele<=0) for sdf_ele in sdf])==0

    def get_gt_sdf(self,scene,point):
        query_point=o3d.core.Tensor(point,dtype=o3d.core.Dtype.Float32) #输入的点一定要是二维的
        sdf=scene.compute_signed_distance(query_point)
        return sdf

    def total_cost(self):
        
        bs,num_agents,horizon,dim=self.traj.shape

        # dis_loss=self.get_dis_loss(self.traj[...,1:])
        dis_loss=self.fast_cal_dis(self.traj[...,1:])

        if self.sdf_network:
            env_collision_loss=self.get_sdf_loss(self.traj[...,1:])
        else:
            env_collision_loss=self.kal_mean_sdf_loss(self.traj)
        
        inter_collision_loss=self.get_inter_collision_loss(num_agents,horizon,self.traj,self.safe_dis,self.device)

        total_loss=env_collision_loss+inter_collision_loss+dis_loss

        return total_loss,env_collision_loss,inter_collision_loss,dis_loss

def start_A_star_optimize(A_star_path,root_path,num_agents,device,lr,num_epoch,mesh_path,sdf_thresh,safe_dis,sdf_network):
    save_path=root_path+'_optimized_path_'+str(num_agents)+'/'
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs(save_path)
    raw_dataset=optimize_dataset(A_star_path)
    optimize_loader=DataLoader(raw_dataset,
                            shuffle=True,
                            batch_size=num_agents,
                            num_workers=4,
                            drop_last=True)
    for id,traj in tqdm(enumerate(optimize_loader)):
        traj=traj[None,...].float().to(device).requires_grad_(True)
        traj_opti=traj_optimizer(traj,lr,num_epoch,mesh_path,safe_dis,device,sdf_network,save_path,sdf_thresh)
        traj_opti.start_optimize(id)

# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--save-to", default="/home/dell/zihan/multi_nt/multi_trajs/A_star_based_4/", type=str)
#     parser.add_argument("--mesh-path", default="/home/dell/zihan/potential_mesh/norm.ply", type=str)
#     parser.add_argument("--raw-data", default="/home/dell/zihan/multi_nt/try_star_traj", type=str)
#     parser.add_argument("--sdf-model-path", default="/home/dell/zihan/multi_nt/final_try_sdf/38.pth", type=str)

#     parser.add_argument("--lr", default=0.01, type=float)
#     parser.add_argument("--sdf-thresh", default=0.031, type=float)
#     parser.add_argument("--safe-dis", default=0.1, type=float)
#     parser.add_argument("--optimize-epoch",default=2500,type=int)
#     parser.add_argument("--batch-size",default=8,type=int)
#     parser.add_argument("--num-workers",default=4,type=int)
#     args = parser.parse_args()

#     sdf_model_path=args.sdf_model_path
#     lr=args.lr
#     num_epoch=args.optimize_epoch
#     mesh_path=args.mesh_path
#     safe_dis=args.safe_dis
#     sdf_thresh=args.sdf_thresh
#     single_raw_data=args.raw_data
#     save_dir=args.save_to

#     sdf_network=transSDF(128)
#     device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     sdf_network.load_state_dict(torch.load(sdf_model_path))
#     sdf_network.to(device).eval()
#     raw_dataset=optimize_dataset(single_raw_data)
#     optimize_loader=DataLoader(raw_dataset,
#                             shuffle=True,
#                             batch_size=args.batch_size,
#                             num_workers=args.num_workers,
#                             drop_last=True)
#     for id,traj in tqdm(enumerate(optimize_loader)):
#         traj=traj[None,...].float().to(device).requires_grad_(True)
#         traj_opti=traj_optimizer(traj,lr,num_epoch,mesh_path,safe_dis,device,sdf_network,save_dir,sdf_thresh=sdf_thresh)
#         traj_opti.start_optimize(id)