import os
import mujoco
import open3d as o3d
import numpy as np
import trimesh
import copy
from utils.GraspEnv import ParallelGraspEnv
from utils.grasp_metric import graspit_measure_hard, gws_pyramid_extension, eplison
from pyquaternion import Quaternion


IN_PLANE_ROTATION_STEPS = 8
DIST_STEPS = 8
DIST_INIT = -0.005
DIST_INTERVAL = 0.0025
INIT_GRIPPER_WIDTH = 0.0255
PARALLEL_GRIPPER = True
if PARALLEL_GRIPPER:
    ANGLE_RANGE = np.pi
    IN_PLANE_ROTATION_STEPS = int(IN_PLANE_ROTATION_STEPS / 2)
else:
    ANGLE_RANGE = 2 * np.pi

ROTATION_START_PLACE = 4
TRANSLATION_START_PLACE = 13
GRIPPER_HEIGHT = 0.02


def get_obj_pose(point, angle_step):

    v1 = point[3:]  # point normal
    v1[1] += 1e-6  # avoid singular value
    # transition = trimesh.transformations.translation_matrix(-pos)
    v0 = np.array([-1, 0, 0])
    angle = trimesh.transformations.angle_between_vectors(v1, v0)
    product = trimesh.transformations.vector_product(v1, v0)
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, product)[:3, :3]

    gamma = float(angle_step) / IN_PLANE_ROTATION_STEPS * ANGLE_RANGE
    Rot_z = trimesh.transformations.rotation_matrix(gamma, [1, 0, 0])[:3, :3]
    R = np.dot(Rot_z, rotation_matrix)
    try:
        quat = trimesh.transformations.quaternion_from_matrix(R)
    except:
        print(v1, angle, product, rotation_matrix)
        print(R)
        exit()
    transition = - np.dot(point[:3], R.T)
    return transition, quat


if __name__ == "__main__":
    object_name = 'obj0'
    point_cloud = o3d.io.read_point_cloud('../asset/mujoco_asset/{}/{}_1024.ply'.format(object_name, object_name))
    sampled_surface_points = np.concatenate([np.asarray(point_cloud.points), np.asarray(point_cloud.normals)], axis=1)

    render_process = False
    annotation_env = ParallelGraspEnv(obj_xml_path='../asset/mujoco_asset/obj_xml/{}.xml'.format(object_name),
                                      obj_name="object",
                                      use_viewer=render_process)

    # score, width, height, depth, rotation_matrix(9) translation(3), Note, set depth to zero for grasp annotation
    shuffle = False
    if shuffle:
        np.random.shuffle(sampled_surface_points)
    grasp_group = np.zeros([len(sampled_surface_points), IN_PLANE_ROTATION_STEPS, DIST_STEPS, 17])
    for idx, point in enumerate(sampled_surface_points):
        # object_mesh = trimesh.load('../raw_object_models/tri/tri.obj')
        # pc = trimesh.PointCloud(point[:3].reshape(-1, 3), colors=(0, 255, 255))
        # scene = trimesh.Scene([object_mesh, pc])
        # scene.show()
        print('{}-th point'.format(idx))

        for angle_step in range(IN_PLANE_ROTATION_STEPS):
            for dist_step in range(DIST_STEPS):
                break_angle = False
                transition, quat = get_obj_pose(point, angle_step)
                distance = DIST_INIT + dist_step * DIST_INTERVAL
                transition += np.array([-distance, 0, 0])
                # model.body_pos[power_drill_id] = transition
                # model.body_quat[power_drill_id] = quat

                # set init gripper status
                qpos = np.array(annotation_env.sim_data.qpos.flat)
                qpos[0] = INIT_GRIPPER_WIDTH
                qpos[1] = INIT_GRIPPER_WIDTH
                qpos[2:5] = transition
                qpos[5:] = quat
                annotation_env.sim_data.ctrl[0] = 255

                qvel = np.array(annotation_env.sim_data.qvel.flat) * 0

                annotation_env.disable_gravity()
                annotation_env.reset_state(qpos, qvel)
                collision = annotation_env.check_collision()
                if collision:
                    break

                for _ in range(100):
                    annotation_env.sim_data.ctrl[0] = 2
                    annotation_env.step_simulation(render=render_process)
                count = 0
                while count < 20000:
                    object_pre_pos, object_pre_quat = annotation_env.get_object_pose(object_name='object')
                    for _ in range(100):
                        # annotation_env.sim_data.ctrl[0] = 2
                        annotation_env.step_simulation(render=render_process)
                        count += 1
                    collision = annotation_env.check_collision_with_fingertip()
                    if not collision:
                        # print('no contact with gripper')
                        break
                    object_cur_pos, object_cur_quat = annotation_env.get_object_pose(object_name='object')
                    tranlation_diff = np.linalg.norm(object_cur_pos-object_pre_pos)
                    quat_diff = Quaternion.distance(Quaternion(object_cur_quat), Quaternion(object_pre_quat))
                    if collision and tranlation_diff < 0.001 and quat_diff < 0.001:
                        # print('achieve stable contact')
                        break

                    # print(count)

                collision = annotation_env.check_collision_with_fingertip()
                if collision:
                    annotation_env.enable_gravity()
                    for _ in range(500):
                        annotation_env.step_simulation(render=render_process)

                collision = annotation_env.check_collision_with_fingertip()
                if collision:
                    # for _ in range(100):
                    #     annotation_env.step_simulation(simulate=False, render=False)
                    object_pos, object_quat = annotation_env.get_object_pose(object_name='object')
                    R = trimesh.transformations.quaternion_matrix(object_quat)
                    t = trimesh.transformations.translation_matrix(object_pos)
                    R_obj = trimesh.transformations.concatenate_matrices(t, R)

                    # contact_points, metric, forces = annotation_env.get_contact_points()
                    grasp_width = annotation_env.sim_data.qpos[0] * 2
                    metric = annotation_env.get_gripper_contact_score(np.linalg.inv(R_obj), grasp_width, vis=False)
                    # print(metric)


                    # calculate normal with face normal
                    # _, _, triangle_id = obj_mesh_copy.nearest.on_surface(contact_points)
                    # normals = -obj_mesh_copy.face_normals[triangle_id]
                    # torques = np.zeros(normals.shape)
                    # forces = normals.copy()
                    # metric = graspit_measure(forces, torques, normals)
                    # print('+++++++', metric)
                    vis = False
                    if vis:
                        object_mesh = trimesh.load('../asset/mujoco_asset/{}/{}.obj'.format(object_name, object_name))
                        obj_mesh_copy = copy.deepcopy(object_mesh)
                        obj_mesh_copy.apply_transform(R_obj)
                        contact_points = np.stack(contact_points, axis=0)
                        forces = np.stack(forces, axis=0)

                        pc = trimesh.PointCloud(contact_points, colors=(0, 255, 255))
                        ray_visualize = trimesh.load_path(
                            np.hstack((contact_points, contact_points + forces/10.0)).reshape(-1, 2, 3)
                        )
                        scene = trimesh.Scene([obj_mesh_copy, pc, ray_visualize])

                        points = np.concatenate([contact_points, forces], axis=1)
                        force_torque = gws_pyramid_extension(obj_mesh_copy, points, forces=np.ones(len(contact_points)))
                        # forces = [item[:3] for item in force_torque]
                        # torques = [item[3:] for item in force_torque]
                        # metric = graspit_measure_hard(force_torque)
                        metric = eplison(force_torque)  # this metric might be more robust given the consideration

                        scene.show()

                    R_gripper = np.linalg.inv(R_obj)
                    grasp_group[idx, angle_step, dist_step, 0] = metric  # score
                    grasp_group[idx, angle_step, dist_step, 1] = np.max(np.abs(annotation_env.sim_data.qpos[:2])) * 2  # width
                    grasp_group[idx, angle_step, dist_step, 2] = GRIPPER_HEIGHT  # height
                    grasp_group[idx, angle_step, dist_step, ROTATION_START_PLACE:TRANSLATION_START_PLACE] = R_gripper[:3, :3].reshape(-1)
                    grasp_group[idx, angle_step, dist_step, TRANSLATION_START_PLACE:16] = R_gripper[:3, 3]

    if not os.path.exists('../grasp_annotation'):
        os.makedirs('../grasp_annotation', exist_ok=True)
    np.save('../grasp_annotation/{}_grasp_group.npy'.format(object_name), grasp_group)
