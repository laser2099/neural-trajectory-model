import open3d as o3d
import numpy as np
def normalize_mesh(mesh_path,save_path,x_range,y_range,z_range):
    mesh=o3d.io.read_triangle_mesh(mesh_path)
    norm_vertex=[]

    x_min=x_range[0]
    y_min=y_range[0]
    x_range=x_range-x_range[0]
    y_range=y_range-y_range[0]
    for ver in mesh.vertices:
        x_recover=(ver[0]-x_min-x_range[0])/(x_range[1]-x_range[0])+x_range[0]
        y_recover=(ver[1]-y_min-y_range[0])/(y_range[1]-y_range[0])+y_range[0]
        z_recover=(ver[2]-z_range[0])/(z_range[1]-z_range[0])+z_range[0]
        norm_vertex.append([x_recover,y_recover,z_recover])
    mesh.vertices=o3d.utility.Vector3dVector(np.array(norm_vertex)) #ver要是二维
    o3d.io.write_triangle_mesh(save_path,mesh)

if __name__=="__main__":
    # x_range=np.array([-35,17])
    # y_range=np.array([-20,20])
    # z_range=np.array([0,12])
    x_range=np.array([-6,6])
    y_range=np.array([-6,6])
    z_range=np.array([0,3])
    mesh_path='/home/dell/zihan/multi_gf/mesh_map.ply'
    save_path='/home/dell/zihan/multi_gf/mesh_map_norm.ply'
    normalize_mesh(mesh_path,save_path,x_range,y_range,z_range)