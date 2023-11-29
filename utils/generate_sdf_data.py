import open3d as o3d
import numpy as np
from tqdm import tqdm

def get_up_down(coords, radius, lbound):
    return  max(coords - radius, lbound),  max(coords + radius, lbound)


def generate_sdf_data(mesh_path,radius, save_path,x_range,y_range,z_range, inner_sample_density = 0.1, outer_sample_density=0.4, lbound=-0.022):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    sample_points = mesh.sample_points_uniformly(number_of_points=12000)

    surface_point = np.asarray(sample_points.points)

    # down_x, up_x = get_up_down(surface_point[:, 0], radius, lbound)
    # down_y, up_y = get_up_down(surface_point[:, 1], radius, lbound)
    # down_z, up_z = get_up_down(surface_point[:, 2], radius, lbound)

    # inner_sample_x = (0 - down_x) * np.random.uniform(down_x, 0,  (0 - down_x) // 0.05)  + down_x # \in [0, 1],  
    # upper_sample_x = (up_x - 0) * np.random.uniform(down_x, 0,  (up_x - 0) // 0.05)  + 0 # \in [0, 1],  

    for i,point in tqdm(enumerate(surface_point)):
        x=np.arange(max((point[0]-radius),x_range[0]),point[0],0.02)
        y=np.arange(max((point[1]-radius),y_range[0]),point[1],0.02)
        z=np.arange(max((point[2]-radius),z_range[0]),point[2],0.01)
        xx,yy,zz=np.meshgrid(x,y,z)
        down_coordinates=np.vstack((xx.flatten(),yy.flatten(),zz.flatten())).T
        # query_point=o3d.core.Tensor(coordinates,dtype=o3d.core.Dtype.Float32)
        # sdf=scene.compute_signed_distance(query_point)
        x_up=np.arange(point[0],min((point[0]+radius),x_range[1]),0.05)
        y_up=np.arange(point[1],min((point[1]+radius),y_range[1]),0.05)
        z_up=np.arange(point[2],min((point[2]+radius),z_range[1]),0.02)
        xx_up,yy_up,zz_up=np.meshgrid(x_up,y_up,z_up)
        up_coordinates=np.vstack((xx_up.flatten(),yy_up.flatten(),zz_up.flatten())).T
        final_coordinates=np.concatenate((down_coordinates,up_coordinates),axis=0)
        
        if i==0:
            out=final_coordinates
        else:
            out=np.concatenate((out,final_coordinates),axis=0)
    # all_uniform_coordinates = np.random.uniform( min_x, max_x, 100000)
    # out = np.concatenate( (outer_coordinates, inner_coordinates, surface_points, all_uniform_coordinates), axis=0)
    x_mean=np.mean(x_range)
    x_std=(x_range[1]-x_range[0])/4
    y_mean=np.mean(y_range)
    y_std=(y_range[1]-y_range[0])/4
    z_mean=np.mean(z_range)
    z_std=(z_range[1]-z_range[0])/4
    x_samples=np.random.normal(x_mean,x_std,500000)
    y_samples=np.random.normal(y_mean,y_std,500000)
    z_samples=np.random.normal(z_mean,z_std,500000)
    samples=np.stack((x_samples,y_samples,z_samples),axis=-1)
    out_coordinates=np.concatenate((out,surface_point,samples),axis=0)
    np.savetxt(save_path,out_coordinates,delimiter=',')



if __name__=='__main__':
    mesh_path='/home/dell/zihan/potential_mesh/norm.ply'
    radius=0.1
    save_path='/home/dell/zihan/multi_nt/sdf_data.csv'
    x_range=[-1.8,0.9]
    y_range=[-1.1,1.1]
    z_range=[0,0.6]
    generate_sdf_data(mesh_path,radius,save_path,x_range,y_range,z_range)
    
