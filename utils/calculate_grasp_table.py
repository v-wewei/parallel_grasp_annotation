import open3d as o3d
import numpy as np
from zy_parallel_gripper_layer import ZYParallelGripperLayer
import torch
import trimesh

from utils.grasp_group import GraspGroup, Grasp


def get_grasp_pose(grasp):
    hand_pose = np.eye(4)
    hand_pose[:3, :3] = grasp[4:13].reshape(3, 3).copy()
    hand_pose[:3, 3] = grasp[13:16].copy()
    return hand_pose


if __name__ == "__main__":
    object_name = 'obj0'
    filename = '../asset/mujoco_asset/{}/{}.obj'.format(object_name, object_name)
    object_mesh = trimesh.load(filename)
    device = 'cuda'
    gripper = ZYParallelGripperLayer(show_mesh=True, make_contact_points=False, to_mano_frame=True, device=device)

    grasp_array_a = np.load('../grasp_annotation/{}_sampled_grasp_group.npy'.format(object_name)).reshape(-1, 17)

    vis = False
    grasp_pairs = []
    count = 0

    grasp_table = np.zeros((len(grasp_array_a), 24, len(grasp_array_a), 24))
    grasp_mask = np.load('../grasp_annotation/{}_sampled_grasp_table.npy'.format(object_name))
    # print(grasp_array_a.shape)

    for idx_a, grasp_a in enumerate(grasp_array_a):
        print(idx_a)
        # [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
        grasp_width = 0.0255 - grasp_a[1] / 2
        theta = np.array([grasp_width, grasp_width], dtype=np.float32)
        theta = torch.from_numpy(theta).to(device)
        # hand_pose = np.eye(4)
        # hand_pose[:3, :3] = grasp_a[4:13].reshape(3, 3)
        # hand_pose[:3, 3] = grasp_a[13:16]
        pose = torch.eye(4).to(device).reshape(-1, 4, 4).float()
        hand_mesh = gripper.get_forward_hand_mesh(pose, theta)[0]

        collision_idx_available = np.where(grasp_mask[idx_a, :, 1]==1)[0]

        grasp_pose_a = get_grasp_pose(grasp_a)
        object_pose = np.linalg.inv(grasp_pose_a)
        for collision_idx_a in collision_idx_available:
            # print(collision_idx_a)
            hand_mesh_a = hand_mesh.copy()
            hand_mesh_a.visual.face_colors = np.array([0, 255, 255])
            angle = np.pi / 12 * collision_idx_a
            rot_matrix = trimesh.transformations.rotation_matrix(angle=angle, direction=np.array([0, 1, 0]))
            T = trimesh.transformations.concatenate_matrices(rot_matrix, object_pose)
            hand_over_pose_a = np.linalg.inv(T)

            hand_mesh_a.apply_transform(hand_over_pose_a)
            # (object_mesh + hand_mesh_a).show()

            collision_manager = trimesh.collision.CollisionManager()
            collision_manager.add_object(name='hand_over_gripper_a', mesh=hand_mesh_a)

            for idx_b, grasp_b in enumerate(grasp_array_a.copy()):
                if idx_b == idx_a:
                    continue
                # if not (idx_b == 41):
                #     continue
                grasp_width = 0.0255 - grasp_b[1] / 2
                theta = np.array([grasp_width, grasp_width], dtype=np.float32)
                theta = torch.from_numpy(theta).to(device)
                # hand_pose = np.eye(4)
                # hand_pose[:3, :3] = grasp_b[4:13].reshape(3, 3)
                # hand_pose[:3, 3] = grasp_b[13:16]
                pose = torch.eye(4).to(device).reshape(-1, 4, 4).float()
                hand_mesh_ = gripper.get_forward_hand_mesh(pose, theta)[0]
                hand_mesh_.visual.face_colors = np.array([255, 0, 255])

                collision_mask = np.array(grasp_mask[idx_b, :, 1], dtype=bool)
                put_angle_mask = np.array(grasp_mask[idx_b, :, 0], dtype=bool)

                mask = put_angle_mask & collision_mask

                valid_idx = np.where(mask == 1)[0]

                grasp_pose_b = get_grasp_pose(grasp_b)
                object_pose_b = np.linalg.inv(grasp_pose_b)
                for idx in valid_idx:
                    # if not (idx == 17):
                    #     continue
                    hand_mesh_b = hand_mesh_.copy()
                    angle = np.pi / 12 * idx
                    rot_matrix = trimesh.transformations.rotation_matrix(angle=angle, direction=np.array([0, 1, 0]))
                    T = trimesh.transformations.concatenate_matrices(rot_matrix, object_pose_b)
                    hand_over_pose_b = np.linalg.inv(T)
                    hand_mesh_b.apply_transform(hand_over_pose_b)
                    # print(idx_a, collision_idx_a, idx_b, idx)
                    # (object_mesh + hand_mesh_a + hand_mesh_b).show()

                    is_collision = collision_manager.in_collision_single(hand_mesh_b)

                    if not is_collision:
                        # print('no collision found')
                        grasp_table[idx_a, collision_idx_a, idx_b, idx] = 1

    np.save('../grasp_annotation/{}_sampled_ho_grasp_table.npy'.format(object_name), grasp_table)