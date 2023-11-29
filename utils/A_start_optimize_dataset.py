import torch
from torch.utils.data import Dataset
import os
import numpy as np

class optimize_dataset(Dataset):
    def __init__(self,data_path) -> None:
        self.path=data_path
        self.file_lst=os.listdir(self.path)

    def __len__(self) -> int:
        return len(self.file_lst)

    def __getitem__(self, index) -> dict:
        file=self.path+'/'+self.file_lst[index]
        data=np.genfromtxt(file,delimiter=',')
        length=[]
        for i in range(len(data)-1):
            current_pos=data[i]
            next_pos=data[i+1]
            dis=self.cal_distance(current_pos,next_pos)
            length.append(dis)
        traj_len=sum(length)
        t=np.linspace(0,traj_len,num=len(data),endpoint=True) #assume v=1
        data_with_t=np.zeros((len(data),4))
        data_with_t[...,1:]=data
        data_with_t[...,0]=t
        return data_with_t
    
    def cal_distance(self,point1,point2):
        distance=np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(point1[2]-point2[2])**2)
        return distance