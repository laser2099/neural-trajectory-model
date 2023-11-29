import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from eval_utils import inter_collision_max_margin_loss, timestamps_to_intervals, interval_to_timestamps


class NeuralTrajsCriterion(_Loss):
    def __init__(self,
            safe_dis_sep,
            mesh_path,
            safe_time_sep,
            sdf_thresh,
            matching_criterion,
            sdf_model, 
            trajs_loss_weights,
            sdf_inner_batch_size = 32,
            matching_weight = 0.7,             
            epsilon:float=1e-4,
            compute_travel_time=False,
            compute_inter_collision=False,
            compute_sdf_loss=True,
            compute_distance_loss=False,
            ) -> None:
        super().__init__()

        self.safe_dis_sep=safe_dis_sep
        self.safe_time_sep=safe_time_sep
        self.sdf_thresh=sdf_thresh
        self.epsilon = epsilon
        self.matching_criterion = matching_criterion
        self.sdf_model = sdf_model
        self.trajs_loss_weights = trajs_loss_weights
        self.matching_weight = matching_weight
        self.sdf_inner_batch_size = sdf_inner_batch_size
        self.compute_travel_time = compute_travel_time
        self.compute_inter_collision = compute_inter_collision
        self.compute_sdf_loss = compute_sdf_loss
        self.compute_distance_loss = compute_distance_loss
        self.mesh_path=mesh_path
        if sdf_model:
            for param in sdf_model.parameters():
                param.required_gradient = False

    def forward(self, model_trajs, gt_trajs, reduce=True):
        return self.model_trajs_loss(
            model_trajs, gt_trajs, 
            self.matching_criterion,
            self.sdf_model, 
            self.trajs_loss_weights,
            self.matching_weight,
        )

    def kal_mean_sdf_loss(self, trajs):
        import kaolin as kal
        from kaolin.ops.mesh import check_sign,index_vertices_by_faces
        import numpy as np
        import open3d as o3d
        # using nvidia kaolin to do differentiable sdf
        # TODO: check correctness

        test_mesh=o3d.io.read_triangle_mesh(self.mesh_path)
        vertices=torch.tensor(np.asarray(test_mesh.vertices)).float().to(trajs.device).unsqueeze(0)
        faces=torch.tensor(np.asarray(test_mesh.triangles),dtype=torch.long).to(trajs.device)
        face_vertices=index_vertices_by_faces(vertices,faces)[0]
        coordinates = trajs[...,1:].view(-1,3)
        
        distance, _, _= kal.metrics.trianglemesh._unbatched_naive_point_to_mesh_distance(coordinates, face_vertices)
        flag = kal.ops.mesh.check_sign(vertices, faces, coordinates.reshape(1,-1,3)).squeeze(0)
        quan_flag = ~flag*2-1
        zeros = torch.zeros([1], device=trajs.device)
        sdf_lst = quan_flag * torch.sqrt(distance)
        sdf_loss = torch.max((self.sdf_thresh - sdf_lst), zeros).mean()
        return sdf_loss        


    def travel_distance(self, trajs):
        # trajs: bsz x num_agents x num_waypoints x 4 [t, x, y, z]
        start_coordinates = trajs[...,:-1,1:]
        end_coordinates = trajs[...,1:,1:]
        pdist = torch.nn.PairwiseDistance(p=2)
        distance = pdist(start_coordinates, end_coordinates)
        traj_distances = distance.sum(-1)
        # return shape: bsz x num_agents x 1 
        return traj_distances

    def travel_time(self, trajs):
        # assume trajs in time-intervals
        # trajs: bsz x num_agents x num_waypoints x 4 [t, x, y, z]
        travel_time = trajs[...,1:,:1].sum(-1) # initial coordinate is the starting timestamp or 0
        # return shape: bsz x num_agents x 1 
        return travel_time        

    def mean_sdf_loss_(self, points3d, sdf_model):
        sdf_lst = sdf_model(points3d)
        zeros = torch.zeros([1], device=sdf_lst.device)
        # using mean to avoid magnitude issue
        sdf_loss = torch.max((self.sdf_thresh - sdf_lst), zeros).mean()
        return sdf_loss
    
    def mean_sdf_loss(self, trajs, sdf_model):
        points3d  = trajs[...,1:].view(-1,3)
        points3d_segs = torch.split(points3d, len(points3d // self.sdf_inner_batch_size + 1))
        loss_list = [self.mean_sdf_loss_(points, sdf_model) for points in points3d_segs]
        return torch.tensor(loss_list).mean()
    
    def trajs_losses(self, trajs, sdf_model):
        # bs, num_agents, horizon, dim = trajs.shape
        mean_distance_loss = self.travel_distance(trajs).mean() if self.compute_distance_loss else None
        mean_travel_time_loss = self.travel_time(trajs).mean() if self.compute_travel_time else None
        if sdf_model !=None:
            # using SDF neural field: 
            env_collision_loss = self.mean_sdf_loss(trajs, sdf_model) if self.compute_sdf_loss else None
        else:
            # using nvidia kaolin:
            env_collision_loss = self.kal_mean_sdf_loss(trajs)
        inter_collision_loss = inter_collision_max_margin_loss(
            interval_to_timestamps(trajs), 
            self.safe_dis_sep, self.safe_time_sep) if self.compute_inter_collision else None
        
        return (
            env_collision_loss, inter_collision_loss, mean_distance_loss, mean_travel_time_loss
        )

    def trajs_criteria(self, 
            loss_weights, 
            env_collision_loss,
            inter_collision_loss, 
            mean_distance_loss, 
            mean_travel_time_loss = None,
        ):
        # loss_weights: sdf_loss, inter_loss, travel_distance_loss, travel_time_loss,
        # bs, num_agents, horizon, dim = trajs.shape

        total_traj_criteria_loss = (
                    (loss_weights[0] * env_collision_loss  if env_collision_loss is not None else 0)
                    + (loss_weights[1] * inter_collision_loss if inter_collision_loss is not None else 0)
                    # TODO: to be normalized
                    + (loss_weights[2] * mean_distance_loss if mean_distance_loss is not None else 0)
                    # TODO: without velocity constraints, travel_time loss is meaningless
                    # TODO: to be normalized
                    + (loss_weights[3] * mean_travel_time_loss if mean_travel_time_loss is not None else 0)
        )
        return total_traj_criteria_loss
    
    def model_trajs_loss(
            self, 
            model_trajs, gt_trajs, 
            matching_criterion,
            sdf_model, 
            trajs_loss_weights,
            matching_weight = 0.7, 
        ):
            bsz, num_agents, num_points, dim = model_trajs.shape
            matching_loss = matching_criterion(model_trajs.view(bsz, -1, 4), gt_trajs.view(bsz, -1, 4))
            env_collision_loss, inter_collision_loss, mean_distance_loss, mean_travel_time_loss = (
                self.trajs_losses(model_trajs, sdf_model)
            )
            trajs_loss = self.trajs_criteria(
                trajs_loss_weights,
                env_collision_loss, 
                inter_collision_loss, 
                mean_distance_loss, 
                mean_travel_time_loss,
            )
            extras = dict(
                matching_loss = matching_loss,
                env_collision_loss = env_collision_loss, 
                inter_collision_loss= inter_collision_loss, 
                mean_distance_loss= mean_distance_loss, 
                mean_travel_time_loss= mean_travel_time_loss, 
            )
            return matching_weight * matching_loss + (1 - matching_weight) * trajs_loss, extras
