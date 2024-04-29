import mujoco
from robosuite.models import MujocoWorldBase
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string
# from robosuite.utils.binding_utils import MjSim
# from mujoco import _functions

# third party
from utils.gripper import CustomGripper
from utils.grasp_metric import graspit_measure

import numpy as np
from pysdf import SDF
import os
import mujoco_viewer
import trimesh
CUR_DIR = os.path.dirname(__file__)


def get_grasps_from_numpy(npy_filepath=None):
    assert npy_filepath
    return np.load(npy_filepath)


class ParallelGraspEnv(object):
    def __init__(self, obj_xml_path=None, obj_name='object',
                 gripper_base_offset=np.array([-0.00, 0.0, 0.0]),
                 gripper_quat=np.array([0.5, 0.5, 0.5, 0.5]),
                 use_viewer=True):
        self.gripper = None
        self.world = MujocoWorldBase()
        self.gripper = self.init_gripper(gripper_base_offset, gripper_quat)
        point_path = os.path.join(CUR_DIR, '../asset/parallel_gripper/points.npy')
        point_transform = trimesh.transformations.quaternion_matrix(gripper_quat)
        point_transform[:3, 3] = gripper_base_offset
        self.gripper_contact_points = trimesh.transformations.transform_points(np.load(point_path), point_transform)
        # self.left_contact_points = gripper_contact_points.copy()
        # self.right_contact_points = gripper_contact_points.copy()
        # self.num_gripper_pad_points = len(gripper_contact_points)

        self.world.merge(self.gripper)
        dynamic_object = self.init_dynamic_object(obj_xml_path, obj_name)
        self.world.merge_assets(dynamic_object)
        self.world.worldbody.append(dynamic_object.get_obj())

        obj_mesh_name = obj_xml_path.split('/')[-1].split('.')[0]
        obj_mesh_path = os.path.join(CUR_DIR, '../asset/objs/{}.obj'.format(obj_mesh_name))
        self.obj_mesh = trimesh.load(obj_mesh_path)
        self.obj_sdf = SDF(self.obj_mesh.vertices, self.obj_mesh.faces)  # (num_vertices, 3) and (num_faces, 3)

        self.sim_model, self.sim_data = self.create_mujoco_sim()
        # self.sim = MjSim(self.world.get_model(mode='mujoco'))

        if use_viewer:
            self.viewer = self.create_viewer()
        else:
            self.viewer = None
        self.save_xml()
        # exit()

    def save_xml(self):
        self.world.save_model('temp_text.xml')
        
    def create_mujoco_sim(self):
        model = self.world.get_model(mode="mujoco")
        data = mujoco.MjData(model)
        return model, data
        # return self.sim.model, self.sim.data_bak

    @staticmethod
    def init_gripper(gripper_base_offset=np.array([-0.09, 0.0, 0.0]), quat=np.array([0.5, 0.5, 0.5, 0.5])):
        gripper = CustomGripper()
        gripper._elements["root_body"].set("pos", array_to_string(gripper_base_offset))
        gripper._elements["root_body"].set("quat", array_to_string(quat))  # w x y z

        return gripper
        
    @staticmethod
    def init_dynamic_object(xml_path, obj_name):
        dynamic_object = MujocoXMLObject(
            fname=xml_path,
            name=obj_name,
            joints=[dict(type="free", damping="0.1")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        return dynamic_object

    def disable_gravity(self):
        self.sim_model.opt.gravity[:] = [0, 0, 0]

    def enable_gravity(self):
        self.sim_model.opt.gravity[:] = [10, 0, 0]

    def create_viewer(self):
        viewer = mujoco_viewer.MujocoViewer(self.sim_model, self.sim_data)
        viewer.vopt.geomgroup[0] = 1
        viewer.vopt.geomgroup[1] = 0
        viewer.vopt.sitegroup[0] = 0
        viewer.vopt.sitegroup[1] = 0
        return viewer

    def get_body_id(self, name="object"):
        body_id = mujoco.mj_name2id(
            self.sim_model,
            mujoco.mjtObj.mjOBJ_BODY,
            name+'_main',
        )
        return body_id

    def get_geom_name(self, id=0):
        geom_name = mujoco.mj_id2name(
            self.sim_model,
            mujoco.mjtObj.mjOBJ_GEOM,
            id,
        )
        return geom_name

    def check_collision(self):
        contacts = self.sim_data.contact[:self.sim_data.ncon]
        collision = True if len(contacts) > 0 else False
        return collision

    def check_collision_with_fingertip(self):
        contacts = self.sim_data.contact[:self.sim_data.ncon]
        gripper_collision = True if len(contacts) > 0 else False

        if gripper_collision:
            left_fingertip_contact = False
            right_fingertip_contact = False

            for i, contact in enumerate(contacts):
                g1, g2 = self.get_geom_name(contact.geom1), self.get_geom_name(contact.geom2)
                if "gripper0_finger_l_tippad" in [g1, g2]:
                    left_fingertip_contact = True
                if "gripper0_finger_r_tippad" in [g1, g2]:
                    right_fingertip_contact = True
            if left_fingertip_contact and right_fingertip_contact:
                return gripper_collision
            else:
                return not gripper_collision
        else:
             return gripper_collision

    def set_state(self, qpos, qvel):
        self.sim_data.qpos[:] = np.copy(qpos)
        self.sim_data.qvel[:] = np.copy(qvel)
        if self.sim_model.na == 0:
            self.sim_data.act[:] = None
        mujoco.mj_forward(self.sim_model, self.sim_data)

    def reset_state(self, qpos, qvel):
        mujoco.mj_resetData(self.sim_model, self.sim_data)
        self.set_state(qpos, qvel)

    def step_simulation(self, simulate=True, render=False):
        if simulate:
            mujoco.mj_step(self.sim_model, self.sim_data)
            # As of MuJoCo 2.0, force-related quantities like cacc are not computed
            # unless there's a force sensor in the model.
            # See https://github.com/openai/gym/issues/1541
            mujoco.mj_rnePostConstraint(self.sim_model, self.sim_data)
        if render:
            self.viewer.render()

    def get_object_pose(self, object_name='object'):
        object_pos = np.copy(self.sim_data.body('{}_main'.format(object_name)).xpos)
        object_ori = np.copy(self.sim_data.body('{}_main'.format(object_name)).xquat)
        return object_pos, object_ori

    def get_contact_points(self, object_name='object'):
        contacts = self.sim_data.contact[:self.sim_data.ncon]
        contact_points = []
        forces = []
        torques = []
        normals = []
        for idx, contact in enumerate(contacts):
            # print(contact)
            # print(contact.geom1, contact.geom2)
            g1, g2 = self.get_geom_name(contact.geom1), self.get_geom_name(contact.geom2)
            if object_name in g1 or object_name in g2:
                contact_points.append(contact.pos)

                # print(self.sim_data.cfrc_ext)
                if object_name in g1:
                    normals.append(-contact.frame[:3])
                    forces.append(-contact.frame[:3])
                    # result = np.zeros([6])
                    # mujoco.mj_contactForce(self.sim_model, self.sim_data, idx, result=result)
                    # print(normals)
                    # print(np.linalg.norm(result))
                if object_name in g2:
                    normals.append(contact.frame[:3])
                    forces.append(contact.frame[:3])
                    # result = np.zeros([6])
                    # mujoco.mj_contactForce(self.sim_model, self.sim_data, idx, result=result)
                    # print(normals)
                    # print(result)
                torques.append([0, 0, 0])
        metric = graspit_measure(forces, torques, normals)
        # print(metric)
        return contact_points, metric, forces#  , forces, torques, normals

    def get_inhand_angle_range(self, object_pose, grasp_width,
                               put_direction=np.array([0, 0, 1.0]),
                               rot_axis=np.array([0, 1, 0])):
        qpos = np.array(self.sim_data.qpos.flat)
        qvel = np.array(self.sim_data.qvel.flat) * 0

        qpos[0] = grasp_width / 2 + 0.0025
        qpos[1] = grasp_width / 2 + 0.0025

        angle_up, angle_down = 0.0, 0.0
        put_angles = -np.ones(24)

        collision_mask = np.zeros(24)

        for i in range(0, 24):
            angle = np.pi / 12 * i
            rot_matrix = trimesh.transformations.rotation_matrix(angle=angle, direction=rot_axis)
            T = trimesh.transformations.concatenate_matrices(rot_matrix, object_pose)
            new_direction = trimesh.transformations.transform_points(put_direction.reshape(-1, 3), T, translate=False)

            put_angle = np.arccos(new_direction[0, 0]) / np.pi * 180

            # if not (put_angle < 20):
            #     collision_mask[i] = 1.0
            #     continue

            put_angles[i] = put_angle
            trans = T[:3, 3]
            quat = trimesh.transformations.quaternion_from_matrix(T)
            qpos[2:5] = trans
            qpos[5:9] = quat
            self.reset_state(qpos, qvel)
            coll = self.check_collision()
            if coll:
                collision_mask[i] = 1.0
                # if put_angle < 20:
                #     for _ in range(100):
                #         self.step_simulation(simulate=False, render=True)


        # for i in range(1, 36):
        #     angle = -np.pi / 36 * i
        #     rot_matrix = trimesh.transformations.rotation_matrix(angle=angle, direction=rot_axis)
        #     T = trimesh.transformations.concatenate_matrices(rot_matrix, object_pose)
        #     trans = T[:3, 3]
        #     quat = trimesh.transformations.quaternion_from_matrix(T)
        #     qpos[2:5] = trans
        #     qpos[5:9] = quat
        #     self.reset_state(qpos, qvel)
        #     res = self.check_collision()
        #     if res:
        #         angle_down = -np.pi / 36 * (i - 1)
        #         # for _ in range(100):
        #         #     self.step_simulation(simulate=False, render=True)
        #         break

        # return angle_up, angle_down
        return put_angles, collision_mask

    def get_gripper_contact_score(self, hand_pose, grasp_width, vis=False):
        left_contact_points = self.gripper_contact_points.copy()
        left_contact_points[:, 1] -= grasp_width / 2.0
        right_contact_points = self.gripper_contact_points.copy()
        right_contact_points[:, 1] += grasp_width / 2.0

        left_contact_points = trimesh.transform_points(left_contact_points, hand_pose)
        right_contact_points = trimesh.transform_points(right_contact_points, hand_pose)

        if vis:
            pc_left = trimesh.PointCloud(left_contact_points, colors=(255, 255, 0))
            pc_right = trimesh.PointCloud(right_contact_points, colors=(255, 0, 255))
            scene = trimesh.Scene([pc_left, pc_right, self.obj_mesh])
            scene.show()
        left_sdf = self.obj_sdf(left_contact_points)
        left_mask = np.abs(left_sdf) < 0.002
        left_contact_score = left_mask.sum() / len(left_sdf)
        right_sdf = self.obj_sdf(right_contact_points)
        right_mask = np.abs(right_sdf) < 0.002
        right_contact_score = right_mask.sum() / len(right_sdf)
        score = np.sqrt(right_contact_score * left_contact_score)
        # print(left_contact_score, right_contact_score)
        return score




















        



