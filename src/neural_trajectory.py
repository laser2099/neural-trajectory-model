from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from group_transformer import GroupGatheredTransformerEncoder
from src_utils import PositionalEmbedder, FC

class TrajTransformer(nn.Module):
    def __init__(self, dim, num_pp=128, coordinate_dim=4, num_head=8, dropout=0.1, pos_embedder=PositionalEmbedder): #D是256，num_pp=128
        super(TrajTransformer, self).__init__()
        self.num_pp = num_pp
        self.point_embedder = pos_embedder(coordinate_dim, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = dim,
            dim_feedforward = dim,
            nhead = num_head,
            dropout = dropout,
            batch_first = True)
        self.transformer = GroupGatheredTransformerEncoder(encoder_layer, num_layers=4)
        self.output_proj = FC(dim, coordinate_dim, False, 'none')

    def forward(self, init_trajs): 
        #lines: bsz x num_agents x trj_len x coordinate_dim (3D or 4D)
        batch_size, num_agents, nwaypoints, coordinate_dim = init_trajs.shape
        waypoints_embedding = self.point_embedder(init_trajs)
        
        

        # bsz x num_agents * waypoits * embed_size
#         waypoints_embedding = waypoints_embedding.reshape(
#             batch_size, num_agents * nwaypoints, -1
#         )

        out_traj_emb = self.transformer(waypoints_embedding)
        path_out = self.output_proj(out_traj_emb)
#         waypoints_out = spath_out.view(batch_size, num_agents, nwaypoints, coordinate_dim)
        # output: bsz x num_agents x trj_len x coordiante_dim 
        return path_out


def example_trajectory_transformer():
  dim = 256
  num_points=128
  num_agents=100
  
  batch_size = 3
  coordinate_dim = 4
  
  network = TrajTransformer(dim, num_points).to(device)
  mock_batch = torch.FloatTensor(batch_size, num_agents, num_points, coordinate_dim).to(device)
  ans = network(mock_batch)
  print(ans.shape)
