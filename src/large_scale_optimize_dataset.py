import torch
from torch.utils.data import Dataset
import os
import numpy as np

class optimize_dataset(Dataset):
    def __init__(self,data_path,point_num=128) -> None:
        self.path=data_path
        self.file_lst=os.listdir(self.path)
        self.num=point_num

    def __len__(self) -> int:
        return len(self.file_lst)

    def __getitem__(self, index) -> dict:
        file_dir=self.path+'/'+self.file_lst[index]
        csv_lst=os.listdir(file_dir)
        gt_traj=np.zeros([len(csv_lst),self.num,4])
        
        for id,file in enumerate(csv_lst):
            file_name=file_dir+'/'+file
            sin_gt_traj=np.genfromtxt(file_name,delimiter=',')
            gt_traj[id]=sin_gt_traj
         
        return gt_traj

    
   
    
    '''
    a=line[:,:-1,:]
    b=line[:,1:,:]
    dis=np.sum(np.sum(np.linalg.norm((a-b),axis=-1),axis=-1))
    '''