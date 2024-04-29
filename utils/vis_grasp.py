import open3d as o3d
import numpy as np

from utils.grasp_group import GraspGroup, Grasp


if __name__ == "__main__":
    SCORE_THRESHOLD = 0.25
    object_name = 'obj0'
    filename = '../asset/mujoco_asset/{}/{}.obj'.format(object_name, object_name)
    o3d_mesh = o3d.io.read_triangle_mesh(filename=filename)
    o3d_mesh.compute_vertex_normals()

    # grasp_array = np.load('../model/models/tri/grasp_label_sim.npy').reshape(-1, 17)
    # grasp_array = np.load('./tri_grasp_group.npy').reshape(-1, 17)
    grasp_array = np.load('../grasp_annotation/{}_grasp_group.npy'.format(object_name)).reshape(-1, 17)

    # grasp_array = np.load('../data_bak/{}_refined_grasp_group.npy'.format(object_name)).reshape(-1, 17)
    # grasp_array.sort_by_score()
    scores = grasp_array[:, 0]
    mask = scores > SCORE_THRESHOLD
    grasp_array = grasp_array[mask]

    # grasp_array[:, 0] = 0.3
    grasp_array[:, 1] += 0.00
    grasp_array[:, 3] += 0.01
    grasp_group = GraspGroup(grasp_array)
    grasp_group.sort_by_score()
    grasp_group = grasp_group.nms(translation_thresh=0.015, rotation_thresh=45.0 / 180.0 * np.pi)
    print(len(grasp_group))

    grippers = []
    for idx in range(len(grasp_group)):
        grasp = grasp_group[idx]
        gripper = grasp.to_open3d_geometry()
        grippers.append(gripper)

    o3d.visualization.draw_geometries([o3d_mesh, *grippers])


