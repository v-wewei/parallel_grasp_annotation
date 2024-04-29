import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
import os


def create_dense_point_cloud(mesh_path, save_path, voxel_size=0.001):
    # os.makedirs(save_path, exist_ok=True)

    print('Read mesh from:', mesh_path)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    n = np.asarray(mesh.vertex_normals)

    f_i, bc = pcu.sample_mesh_poisson_disk(v, f, num_samples=-1, radius=0.0005, use_geodesic_distance=True)
    # print(n.shape, bc.shape, f_i.shape)
    v_poisson = pcu.interpolate_barycentric_coords(f, f_i, bc, v)
    n_poisson = pcu.interpolate_barycentric_coords(f, f_i, bc, n)

    object_cloud = o3d.geometry.PointCloud()
    object_cloud.points = o3d.utility.Vector3dVector(v_poisson)
    object_cloud.normals = o3d.utility.Vector3dVector(n_poisson)
    object_cloud = object_cloud.voxel_down_sample(voxel_size)

    o3d.io.write_point_cloud(save_path, object_cloud)
    # np.savez(save_path, points=v_poisson, normals=n_poisson)


if __name__ == '__main__':
    src_dir = '../asset/objs/'
    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            if not filename.endswith('.obj'):
                continue
            if "_simplified" in filename:
                continue
            file_path = os.path.join(root, filename)
            print(file_path)
            create_dense_point_cloud(file_path, os.path.join(src_dir, filename.replace('.obj', '.ply')))