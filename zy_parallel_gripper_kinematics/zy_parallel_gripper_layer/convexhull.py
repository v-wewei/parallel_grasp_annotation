import trimesh
import os
import numpy as np
from mesh_to_sdf import get_surface_point_cloud
from scipy.spatial import KDTree
import point_cloud_utils as pcu
import trimesh.sample


def save_part_convex_hull_mesh(dst='../assets/hand_meshes'):
    for root, dirs, files in os.walk(dst):
        for filename in files:
            if filename.endswith('.stl') or filename.endswith('.STL') or filename.endswith('.obj'):
                filepath = os.path.join(root, filename)
                mesh = trimesh.load_mesh(filepath)
                convex_mesh = mesh.convex_hull
                # key = filename.split('.')[0]
                # if key == 'base_link':
                #     mesh_ = mesh.convex_decomposition(
                #         maxConvexHulls=16 if key == 'base_link' else 2,
                #         resolution=800000 if key == 'base_link' else 1000,
                #         minimumVolumePercentErrorAllowed=0.1 if key == 'base_link' else 10,
                #         maxRecursionDepth=10 if key == 'base_link' else 4,
                #         shrinkWrap=True, fillMode='flood', maxNumVerticesPerCH=32,
                #         asyncACD=True, minEdgeLength=2, findBestPlane=False
                #     )
                #     convex_mesh = np.sum(mesh_)
                # else:
                #     convex_mesh = mesh.convex_hull

                if not os.path.exists(dst.replace('hand_meshes', 'hand_meshes_cvx')):
                    os.makedirs(dst.replace('hand_meshes', 'hand_meshes_cvx'))
                new_filepath = filepath.replace('hand_meshes', 'hand_meshes_cvx')
                print('save to path:', new_filepath)
                convex_mesh.export(new_filepath)
    print('save finished')


def sample_points_on_mesh(src_dir='../assets/hand_meshes_cvx'):
    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            print('Sample points on', filename)
            filepath = os.path.join(root, filename)
            convex_mesh = trimesh.load_mesh(filepath)
            # key = filename.split('.')[0]
            # mesh_ = mesh.convex_decomposition(
            #     maxConvexHulls=16 if key == 'base_link' else 2,
            #     resolution=800000 if key == 'base_link' else 1000,
            #     minimumVolumePercentErrorAllowed=0.1 if key == 'base_link' else 10,
            #     maxRecursionDepth=10 if key == 'base_link' else 4,
            #     shrinkWrap=True, fillMode='flood', maxNumVerticesPerCH=32,
            #     asyncACD=True, minEdgeLength=2, findBestPlane=False
            # )
            # convex_mesh = np.sum(mesh_)
            np.random.seed(0)
            points, idx = trimesh.sample.sample_surface_even(convex_mesh, 25000, radius=None)
            point_normals = convex_mesh.face_normals[idx]
            vis = False
            if vis:
                pc = trimesh.PointCloud(points, colors=(255, 255, 0))
                ray_visualization = trimesh.load_path(np.hstack((points,
                                                                points + point_normals / 100)).reshape(-1, 2, 3))
                scene = trimesh.Scene([pc, ray_visualization])
                scene.show()

            info = np.concatenate([points, point_normals], axis=-1)
            dst_dir = src_dir.replace('hand_meshes_cvx', 'hand_points')
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
            np.save(os.path.join(dst_dir, filename.replace('.stl', '.npy').replace('.STL', '.npy').replace('.obj', '.npy')), info)


def sample_visible_points():
    count = 0
    import scipy
    hand = trimesh.load('../assets/hand.obj', force='mesh')
    result = get_surface_point_cloud(hand, scan_count=100, scan_resolution=400)
    points = np.array(result.points)
    # pc = trimesh.PointCloud(points, colors=(255, 255, 0))
    # pc.show()
    point_tree = KDTree(data=points)

    dst = '../assets/hand_composite_points'

    for root, dirs, files in os.walk(dst):
        for filename in files:
            filepath = os.path.join(root, filename)
            point_info = np.load(filepath)
            dist, index = point_tree.query(point_info[:, :3], k=1)
            mask = dist < 0.001

            visible_points = point_info[mask]
            v_sampled = pcu.downsample_point_cloud_on_voxel_grid(0.003, visible_points[:, :3])
            count += len(v_sampled)
            # pc = trimesh.PointCloud(v_sampled, colors=(255, 255, 0))
            # pc.show()
            # exit()
            _, v_sampled_idx = KDTree(point_info[:, :3]).query(v_sampled, k=1)
            # pc = trimesh.PointCloud(point_info[:, :3][v_sampled_idx])
            # pc.show()
            # exit()
            if not os.path.exists('../assets/visible_point_indices'):
                os.makedirs('../assets/visible_point_indices', exist_ok=True)
            np.save('../assets/visible_point_indices/{}.npy'.format(filename[:-4]), v_sampled_idx)
    print(count)


if __name__ == "__main__":
    # sample_points_on_mesh(src_dir='../assets/hand_meshes')
    # sample_points_on_mesh(src_dir='../assets/leap_hand_composite')
    # sample_visible_points()
    save_part_convex_hull_mesh()