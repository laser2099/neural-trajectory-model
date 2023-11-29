import torch

def interval_to_timestamps(trajs, inplace=False):
    if not inplace:
        trajs = trajs.clone()
    trajs[...,0] = trajs[...,0].cumsum(dim=-1)
    return trajs

def timestamps_to_intervals(trajs, inplace=False):
    if not inplace:
        trajs = trajs.clone()
    trajs[..., 1:,0] = trajs[..., 1:,0] - trajs[..., 0:-1,0]
    return trajs

def collision_tensor(trajs, safe_dis, safe_time_sep):
    bsz, num_agents, traj_len, dim = trajs.shape
    trajs = trajs.reshape(bsz, num_agents * traj_len, dim)
    # assume trajs[bsz, num_agent, traj_len, [time, x, y, z]]
    distance = torch.cdist(trajs[:, :, 1:], trajs[:, :, 1:], p=2)
    time_diff = torch.cdist(trajs[:,:, :1], trajs[:,:, :1], p=1)
#     print('time diff:', time_diff.shape)
    
    collision = torch.logical_and(distance < safe_dis, time_diff < safe_time_sep)
    collision = collision.reshape(bsz, num_agents, traj_len, num_agents, traj_len)
    
    # set the collision between waypoints inside an agent's trajectory to be false
    diag_indices = torch.arange(num_agents, device=collision.device)
    collision[:, diag_indices, :, diag_indices, :] = 0
    return collision

def inter_collision_rate(trajs, safe_dis, safe_time_sep):
    bsz, num_agents, traj_len, dim = trajs.shape
    collision = collision_tensor(trajs, safe_dis, safe_time_sep)
    # per waypoint collision
    collision_per_waypoint = collision.view(bsz, num_agents, traj_len, -1).any(-1)
    collision_per_traj = collision_per_waypoint.float().mean(-1)
    # output shape: bsz x num_agents i.e. bsz x num_trajs
    # output per trajectory collision rate = num_of_sampled_waypoints_with_collision / num_sampled_waypoints per trajectory
    return collision_per_traj

def mean_inter_collision_rate(trajs, safe_dis, safe_time_sep):
    collision_per_traj = inter_collision_rate(trajs, safe_dis, safe_time_sep)
    # output shape: bsz 
    # output mean collision rate per multi-agent case instance = mean of per trajectory collision rates
    return collision_per_traj.mean(-1)

def inter_collision_max_margin_loss(
    trajs, 
    safe_dis: float, 
    safe_time_sep: float, 
#     distance_time_lamda:float=0.5,
    epsilon:float=1e-4,
):
    bsz, num_agents, traj_len, dim = trajs.shape
    # element: [time-stamp, x, y, z]
    trajs = trajs.reshape(bsz, num_agents * traj_len, dim)
    distance = torch.cdist(trajs[:, :, 1:], trajs[:, :, 1:], p=2)
    time_diff = torch.cdist(trajs[:,:, :1], trajs[:,:, :1], p=1)

    zeros = torch.zeros([1], device=distance.device)
    distance_margin_loss = torch.max(zeros - epsilon, safe_dis - distance)
    time_sep_margin_loss = torch.max(zeros - epsilon, safe_time_sep - time_diff)
    
#     loss = distance_time_lamda * distance_margin_loss + (1 - distance_time_lamda) * time_sep_margin_loss
    # to optimize towards both distance and time are -epsilon less than their corresponding margin/safe distance
    loss = torch.max(distance_margin_loss, time_sep_margin_loss)
    return loss.mean()

