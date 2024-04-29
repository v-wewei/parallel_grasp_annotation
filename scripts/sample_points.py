import torch
import pickle
import trimesh
import open3d as o3d
import numpy as np
# from lxml import etree as ET
# from pytorch3d.ops import sample_farthest_points
import os

mesh_dir = "/home/v-wewei/code/parallel_grasp_annotation/asset/models"
pcd_dir = '/home/v-wewei/code/parallel_grasp_annotation/asset/raw_pcl'

obj_pcl_buf = {}

'''
Gen point cloud
'''
pointcloud_save_dir = "pcd"
rates = [512] #[200, 400, 600, 800, 1000]

def mesh_to_pointcloud(mesh, box_num_rate,  model_savepath,  min_num=5e1, max_num=5e4):
    # vertices = np.array(mesh.vertices)
    # max_x = vertices[:,0].max()
    # max_y = vertices[:,1].max()
    # max_z = vertices[:,2].max()
    # min_x = vertices[:,0].min()
    # min_y = vertices[:,1].min()
    # min_z = vertices[:,2].min()
    # vol = (max_x - min_x) * (max_y - min_y) * (max_z - min_z) # the volume of AABB
    # print(vol)
    # target_num = int( vol*1e6 / box_num_rate) # !!!
    #
    # if target_num < min_num:
    #     target_num = int(min_num)
    #
    # if target_num > max_num:
    #     target_num = int(max_num)

    target_num = 512
    # print( "Vertex Num: %s, Target Num: %s Vol_Num Rate:%d" % (len(vertices), target_num, box_num_rate) )

    pointcloud = mesh.sample_points_poisson_disk(number_of_points=target_num, use_triangle_normal=True)
    pc = np.array(pointcloud.points)
    print(pc.shape)
    # pointcloud.points = o3d.utility.Vector3dVector(pc)
    ply_path = os.path.join(model_savepath, 'ply')
    pcd_path = os.path.join(model_savepath, 'pcd')
    if not os.path.exists(ply_path):
        os.makedirs(ply_path)
    if not os.path.exists(pcd_path):
        os.makedirs(pcd_path)
    o3d.io.write_point_cloud( os.path.join(ply_path, "%s.ply" % box_num_rate), pointcloud, write_ascii=True)
    o3d.io.write_point_cloud( os.path.join(pcd_path, "%s.pcd" % box_num_rate), pointcloud, write_ascii=True)


generate_pcd = True
if generate_pcd:
    for object in os.listdir(mesh_dir):
        # object = 'gd_spray_can_poisson_000'
        object_code = object.split('.')[0]
        mesh_path = os.path.join(mesh_dir, f"{object_code}.obj")
        mesh = trimesh.load(mesh_path, process=False)

        save_path = os.path.join(mesh_dir.replace("models","raw_pcl"), object_code)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        mesh.export(os.path.join(save_path, 'normalized.ply'), file_type='ply')
        mesh = o3d.io.read_triangle_mesh(os.path.join(save_path, 'normalized.ply'))
        mesh = mesh.compute_vertex_normals()
        model_url = os.path.join(save_path, 'normalized.ply')
        o3d.io.write_triangle_mesh(model_url, mesh)

        model_savepath = os.path.join(mesh_dir.replace("models","raw_pcl"), object_code)
        mesh = o3d.io.read_triangle_mesh(model_url)
        for box_num_rate in rates:
            mesh_to_pointcloud(mesh, box_num_rate, model_savepath)

# '''
# Gen all pointcloud buffer
# '''
# def farthest_point_sample(xyz, npoint, device):
#     """
#     Input:
#         xyz: pointcloud data_bak, [B, N, 3]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#     B, N, C = xyz.size()
#     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
#     distance = torch.ones(B, N).to(device) * 1e10
#     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
#     batch_indices = torch.arange(B, dtype=torch.long).to(device)
#     for i in range(npoint):
#         centroids[:, i] = farthest
#         centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
#         dist = torch.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = torch.max(distance, -1)[1]
#     return centroids
#
# def index_points(points, idx, device):
#     """
#     Input:
#         points: input points data_bak, [B, N, C]
#         idx: sample index data_bak, [B, S]
#     Return:
#         new_points:, indexed points data_bak, [B, S, C]
#     """
#     B = points.size()[0]
#     view_shape = list(idx.size())
#     view_shape[1:] = [1] * (len(view_shape) - 1)
#     repeat_shape = list(idx.size())
#     repeat_shape[0] = 1
#     batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
#     new_points = points[batch_indices, idx, :]
#     return new_points

def gen_pcl_buffer(object, pcd_path):
    object_pcl = o3d.io.read_point_cloud(pcd_path)
    object_pcl_points = torch.tensor(np.asarray(object_pcl.points), device="cuda:0", dtype=torch.float32).reshape(1,-1,3)
    total_object_pcl_numer = object_pcl_points.shape[0]
    sampled_pcl = sample_farthest_points(object_pcl_points, K=4096)[0]
    # sampled_point_idxs = farthest_point_sample(object_pcl_points, 256, device="cuda:0")
    # sampled_pcl = index_points(object_pcl_points, sampled_point_idxs, device="cuda:0")
    obj_pcl_buf[object] = sampled_pcl.reshape(-1,3).cpu().numpy()
    print(total_object_pcl_numer)

generate_pickle = False
if generate_pickle:
    for object in os.listdir(pcd_dir):
        pcd_path = os.path.join(pcd_dir, object, 'pcd/600.pcd')
        gen_pcl_buffer(object, pcd_path)

    with open('../data_bak/pcl_buffer_4096_all.pkl', 'wb') as f:
        pickle.dump(obj_pcl_buf, f)