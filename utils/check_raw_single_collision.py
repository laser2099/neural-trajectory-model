import open3d as o3d
import os
import numpy as np
import shutil
from tqdm import tqdm
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--save-to", default="/home/dell/zihan/multi_nt/A_star_trajs", type=str)
parser.add_argument("--mesh-path", default="/home/dell/zihan/potential_mesh/norm.ply", type=str)
parser.add_argument("--raw-data", default="/home/dell/zihan/multi_nt/try_star_traj/", type=str)
parser.add_argument("--sdf-model-path", default="/home/dell/zihan/multi_nt/final_try_sdf/38.pth", type=str)
args = parser.parse_args()

traj_dir=args.save_to
raw_single_dir=args.raw_data
mesh_path=args.mesh_path

traj_lst=os.listdir(traj_dir)
for file in tqdm(traj_lst):
    csv_file=traj_dir+'/'+file
    points=np.genfromtxt(csv_file,delimiter=',')
    mesh=o3d.io.read_triangle_mesh(mesh_path)
    mesh=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene=o3d.t.geometry.RaycastingScene()
    _=scene.add_triangles(mesh)
    query_point=o3d.core.Tensor(points,dtype=o3d.core.Dtype.Float32) 
    sdf=scene.compute_signed_distance(query_point)
    if all(sdf>0):
        shutil.move(csv_file,raw_single_dir)