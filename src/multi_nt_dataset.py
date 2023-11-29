from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd

class multi_nt_dataset(Dataset):
    def __init__(self,data_path,num_points,x_range,y_range,z_range,t_range) -> None:
        self.path=data_path
        self.num=num_points
        self.file_lst=os.listdir(self.path)
        self.x_range=x_range
        self.x_min=x_range[0]
        self.y_range=y_range
        self.y_min=y_range[0]
        self.z_range=z_range
        self.t_range=t_range

    def __len__(self) -> int:
        return len(self.file_lst)

    def __getitem__(self, index) -> dict:
        file_dir=self.path+'/'+self.file_lst[index]
        csv_lst=os.listdir(file_dir)
        gt_traj=np.zeros([len(csv_lst),self.num,4])
        start=np.zeros([len(csv_lst),4])
        end=np.zeros([len(csv_lst),4])
        line=np.zeros([len(csv_lst),self.num,4])
        for id,file in enumerate(csv_lst):
            file_name=file_dir+'/'+file
            sin_gt_traj,start_pos,end_pos,sin_line=self.read_data(file_name)
            gt_traj[id]=sin_gt_traj
            start[id]=start_pos
            end[id]=end_pos
            line[id]=sin_line
        element={
            "start":start,
            "end":end,
            "line":line,
            "gt_traj":gt_traj
        }
        return element
    
    def straight_line(self,start,end,num_points,time_stamp):
        line=np.zeros([num_points,4])
        line[:,0]=time_stamp
        for i in range(num_points):
            point=start+(end-start)/(num_points-1)*i
            line[i,1:]=point
        return line
    
    def read_data(self,csv_name):
        raw_gt_traj=np.genfromtxt(csv_name,delimiter=',')
        gt_traj=raw_gt_traj.copy()
        self.x_range=self.x_range-self.x_range[0]
        self.y_range=self.y_range-self.y_range[0]
        gt_traj[:,0]=(gt_traj[:,0]-self.t_range[0])/(self.t_range[1]-self.t_range[0])+self.t_range[0]
        gt_traj[:,1]=(gt_traj[:,1]-self.x_min-self.x_range[0])/(self.x_range[1]-self.x_range[0])+self.x_range[0]
        gt_traj[:,2]=(gt_traj[:,2]-self.y_min-self.y_range[0])/(self.y_range[1]-self.y_range[0])+self.y_range[0]
        gt_traj[:,3]=(gt_traj[:,3]-self.z_range[0])/(self.z_range[1]-self.z_range[0])+self.z_range[0]
        start_point=gt_traj[0][1:]
        end_point=gt_traj[-1][1:]
        num_points=gt_traj.shape[0]
        gt_time_stamp=gt_traj[:,0]

        time_stamp_first=[0]
        time_stamp_add=[gt_time_stamp[i+1]-gt_time_stamp[i] for i in range(len(gt_time_stamp)-1)]
        time_stamp_first.extend(time_stamp_add)
        time_stamp=np.array(time_stamp_first)
        gt_traj[:,0]=time_stamp

        line=self.straight_line(start_point,end_point,num_points,time_stamp)
        start=gt_traj[0]
        end=gt_traj[-1]
        return gt_traj,start,end,line
    
