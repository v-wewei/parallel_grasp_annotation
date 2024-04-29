import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

from utils.GraspEnv import ParallelGraspEnv
from utils.grasp_group import GraspGroup


def grasp_array_to_dict(grasp_array):
    grasp_dicts = []
    for grasp in grasp_array:
        grasp_rot = grasp[4:13].reshape(3, 3)
        grasp_quat = trimesh.transformations.quaternion_from_matrix(grasp_rot)
        grasp_dict = {'x': grasp[13], 'y': grasp[14], 'z': grasp[15],
                      'qx': grasp_quat[1], 'qy': grasp_quat[2], 'qz': grasp_quat[3], 'qw': grasp_quat[0],
                      'width': grasp[1]}

        grasp_dicts.append(grasp_dict)
    return grasp_dicts


if __name__ == "__main__":
    '''
    qpos is an array that holds all the joint position values for all joints in the model.

    For normal robot joints (e.g. hinges, sliders), there's one entry per joint. 
    For free joints (the "joint" between a free floating object and the world), there are 7 entries: 
    3 for the position and 4 for the quaternion representing the object's orientation, 
    and the order in the array is the same order that they're specified in the MuJoCo model XML.

    ctrl is an array that holds the control commands for the robot. 
    The length of the array is the number of actuators in the model.
    '''
    SCORE_THRESHOLD = 0.25
    vis_list = []

    object_name = 'obj0'

    put_direction_dict = {'obj1': np.array([0, -1, 0]),
                          'obj0': np.array([0, 0, 1]),
                          'obj2': np.array([0, 0, 1])}

    grasp_array = np.load('../grasp_annotation/{}_grasp_group.npy'.format(object_name)).reshape(-1, 17)

    scores = grasp_array[:, 0]
    mask = scores > SCORE_THRESHOLD
    grasp_array = grasp_array[mask]
    grasp_array[:, 1] += 0.004
    grasps = GraspGroup(grasp_array)
    grasps.sort_by_score()
    grasps = grasps.nms(translation_thresh=0.015, rotation_thresh=45.0 / 180.0 * np.pi).grasp_group_array

    render_process = False
    annotation_env = ParallelGraspEnv(obj_xml_path='../asset/mujoco_asset/obj_xml/{}.xml'.format(object_name),
                                      obj_name="object",
                                      gripper_base_offset=np.array([-0.0, 0, 0.00]),
                                      gripper_quat=np.array([0.5, 0.5, 0.5, 0.5]),
                                      use_viewer=render_process)

    put_direction = put_direction_dict[object_name]
    if 'put_direction' in vis_list:
        obj_mesh = trimesh.load('../asset/mujoco_asset/{}/{}.obj'.format(object_name, object_name))
        origin = np.array([0, 0, 0])
        put_direction_vis = np.column_stack((origin, origin + (put_direction * .05)))
        path = trimesh.load_path(put_direction_vis.reshape((-1, 2, 3)))
        scene = trimesh.Scene([obj_mesh, path])
        scene.show()

    # grasp_array = np.zeros(17)
    grasp_group = []
    grasp_dicts = {
        "attributes": {
            'weight': 45
        },
        "grasp_poses": [],
    }

    count = 0
    grasp_table = np.zeros((len(grasps), 24, 2))

    for idx, grasp in enumerate(grasps):
        # setup a transformation matrix
        depth = grasp[3]
        T = np.eye(4)
        T[:3, :3] = grasp[4:13].reshape(3, 3)
        T[:3, 3] = grasp[13:16]  # + T[:3, 0] * (depth - 0.01)
        T_inv = np.linalg.inv(T)

        rotation = R.from_matrix(T_inv[:3, :3])
        quat = rotation.as_quat()  # [x, y, z, w]

        # set init object and gripper status
        INIT_GRIPPER_WIDTH = grasp[1] / 2.0 + 0.004
        transition = T_inv[:3, 3]
        qpos = np.array(annotation_env.sim_data.qpos.flat)

        # qpos[0] = INIT_GRIPPER_WIDTH
        # qpos[1] = 0.0
        # qpos[2] = INIT_GRIPPER_WIDTH
        # qpos[3] = 0.0
        # qpos[4:7] = transition
        # qpos[7:11] = quat[[3, 0, 1, 2]]

        qpos[0] = INIT_GRIPPER_WIDTH
        qpos[1] = INIT_GRIPPER_WIDTH
        qpos[2:5] = transition
        qpos[5:9] = quat[[3, 0, 1, 2]]

        qvel = np.array(annotation_env.sim_data.qvel.flat) * 0

        # NOTE: disable gravity should be set before reset simulation state to keep simulation correctly
        annotation_env.disable_gravity()
        annotation_env.reset_state(qpos, qvel)

        refine_grasp = False
        if refine_grasp:
            for _ in range(500):  # 1000
                annotation_env.sim_data.ctrl[0] = 0
                annotation_env.step_simulation(simulate=True, render=render_process)

            # collision_after_grasp = annotation_env.check_collision()
            collision = annotation_env.check_collision_with_fingertip()
            if not collision:
                continue

            annotation_env.enable_gravity()
            for _ in range(500):  # 500
                annotation_env.sim_data.ctrl[0] = 0
                annotation_env.step_simulation(simulate=True, render=render_process)
            collision = annotation_env.check_collision_with_fingertip()
        else:
            # for _ in range(250):  # 1000
            #     annotation_env.sim_data.ctrl[0] = 0
            #     annotation_env.step_simulation(simulate=True, render=render_process)
            #
            # collision = annotation_env.check_collision_with_fingertip()
            # if not collision:
            #     continue
            collision = True

        if collision:
            count += 1
            # for _ in range(250):  # 1000
            #     annotation_env.step_simulation(simulate=True, render=render_process)
            # print('collision:', collision)
            if refine_grasp:
                grasp_width = annotation_env.sim_data.qpos[0] * 2
            else:
                grasp_width = grasp[1]
            # object_pos, object_ori = annotation_env.get_object_pose(object_name='object')
            # T = np.eye(4)
            # T[:3, :3] = R.from_quat(object_ori[[1, 2, 3, 0]]).as_matrix()
            # T[:3, 3] = object_pos
            # grasp_pose = np.linalg.inv(T)
            # grasp_translation = grasp_pose[:3, 3]
            # grasp_quat = R.from_matrix(grasp_pose[:3, :3]).as_quat()

            grasp_translation = annotation_env.sim_data.sensordata[:3].ravel().copy().reshape(3, )
            grasp_quat = annotation_env.sim_data.sensordata[3:7].ravel().copy().reshape(4, )

            grasp_pose = trimesh.transformations.quaternion_matrix(grasp_quat)
            grasp_pose[:3, 3] = grasp_translation
            object_pose = np.linalg.inv(grasp_pose)

            put_angles, collision_mask = annotation_env.get_inhand_angle_range(object_pose, grasp_width,
                                                                               put_direction=put_direction,
                                                                               rot_axis=np.array([0, 1, 0]))

            mask_angle = put_angles < 20
            grasp_table[idx, :, 0] = mask_angle

            mask_collision = collision_mask < 1.0
            grasp_table[idx, :, 1] = mask_collision

            mask = mask_angle & mask_collision
            mask_sum = mask.sum()

            nonCollisionList = np.where(mask_collision == 1)[0]
            availableAngleList = np.where(mask_angle == 1)[0]

            grasp_dict = {
                'nonCollisionList': nonCollisionList,
                'availablePutList': availableAngleList,
                'availableList': np.where(mask == 1)[0],
                'openDistance': grasp_width,
                "shiftHand": False,
                'position': {'x': grasp_translation[0], 'y': grasp_translation[1], 'z': grasp_translation[2]},
                'rotation': {'qx': grasp_quat[1], 'qy': grasp_quat[2], 'qz': grasp_quat[3], 'qw': grasp_quat[0]},
                'putAngles': put_angles,
                'collisionMask': collision_mask
            }

            if mask_sum == 0:
                grasp_dict['shiftHand'] = True

            grasp_dicts['grasp_poses'].append(grasp_dict)

            # [score, width, height, depth, rotation_matrix(9), translation(3), object_id]
            grasp_array = np.array(
                [grasp[0], grasp_width, 0.02, 0.0, *(grasp_pose[:3, :3].reshape(-1)), *grasp_pose[:3, 3], 0])
            grasp_group.append(grasp_array)

    print(len(grasp_dicts['grasp_poses']))

    # assert len(grasp_dicts['grasp_poses']) == len(grasps)

    np.save('../grasp_annotation/{}_sampled_grasp_dict.npy'.format(object_name), grasp_dicts)
    grasp_group = np.stack(grasp_group)
    np.save('../grasp_annotation/{}_sampled_grasp_group.npy'.format(object_name), grasp_group)
    np.save('../grasp_annotation/{}_sampled_grasp_table.npy'.format(object_name), grasp_table)
