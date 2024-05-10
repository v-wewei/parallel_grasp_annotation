# hand_meshes layer for torch
import torch
import math
import trimesh
import glob
import os
import numpy as np
import copy
import pytorch_kinematics as pk
from zy_parallel_gripper_layer.convexhull import save_part_convex_hull_mesh, sample_points_on_mesh, sample_visible_points
import roma
# from speed_hand_layer.convexhull import save_part_convex_hull_mesh, sample_points_on_mesh, sample_visible_points


GRIPPER_MAX_WIDTH = 0.0255

BASE_DIR = os.path.split(os.path.abspath(__file__))[0]
# All lengths are in mm and rotations in radians


def save_to_mesh(vertices, faces, output_mesh_path=None):
    assert output_mesh_path is not None
    with open(output_mesh_path, 'w') as fp:
        for vert in vertices:
            fp.write('v %f %f %f\n' % (vert[0], vert[1], vert[2]))
        for face in faces+1:
            fp.write('f %d %d %d\n' % (face[0], face[1], face[2]))
    print('Output mesh save to: ', os.path.abspath(output_mesh_path))


class ZYParallelGripperLayer(torch.nn.Module):
    def __init__(self, to_mano_frame=True, show_mesh=False, make_contact_points=False, device='cuda'):
        super().__init__()
        # for first time run to generate contact points on the hand, set the self.make_contact_points=True
        self.make_contact_points = make_contact_points
        self.show_mesh = show_mesh
        self.device = device

        urdf_path = os.path.join(BASE_DIR, '../urdf/zy_parallel_gripper.urdf')
        self.chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(device=device)
        # print(self.chain)
        # print(self.chain.get_links())
        self.link_dict = {}
        for link in self.chain.get_links():
            if link.name == 'world':
                continue
            self.link_dict[link.name] = link.visuals[0].geom_param[0].split('/')[-1]

        self.to_mano_transform = torch.eye(4).to(torch.float32).to(device)
        if to_mano_frame:
            self.to_mano_transform[:3, :3] = roma.unitquat_to_rotmat(torch.tensor([0.5, 0.5, 0.5, 0.5]))
            self.to_mano_transform[:3, 3] = torch.tensor([0.0, 0, -0.0])

        self.register_buffer('base_2_world', self.to_mano_transform)
        if not (os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_meshes_cvx')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_points')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_composite_points')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/visible_point_indices')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand.obj')
                and os.path.exists(os.path.abspath(os.path.dirname(__file__)) + '/../assets/hand_all_zero.obj')
        ):
            self.create_assets()
        self.meshes = self.load_meshes()

    def create_assets(self):
        '''
        To create needed assets for the first running.
        Should run before first use.
        '''
        self.to_mano_transform = torch.eye(4).to(torch.float32).to(device)
        pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
        theta = np.ones((1, 2), dtype=np.float32) * GRIPPER_MAX_WIDTH

        save_part_convex_hull_mesh()
        sample_points_on_mesh()

        show_mesh = self.show_mesh
        make_contact_points = self.make_contact_points
        self.show_mesh = True
        self.make_contact_points = True

        self.meshes = self.load_meshes()
        mesh = self.get_forward_hand_mesh(pose, theta)[0]
        mesh.export(os.path.join(BASE_DIR, '../assets/hand.obj'))

        self.show_mesh = True
        self.make_contact_points = False
        self.meshes = self.load_meshes()
        mesh = self.get_forward_hand_mesh(pose, theta)[0]
        mesh.export(os.path.join(BASE_DIR, '../assets/hand_all_zero.obj'))

        self.show_mesh = False
        self.make_contact_points = True
        self.meshes = self.load_meshes()

        self.get_forward_vertices(pose, theta)      # SAMPLE hand_composite_points
        sample_visible_points()

        self.show_mesh = True
        self.make_contact_points = False
        self.to_mano_transform[:3, :3] = roma.unitquat_to_rotmat(torch.tensor([0.5, 0.5, 0.5, 0.5]))
        self.to_mano_transform[:3, 3] = torch.tensor([0.0, 0, -0.0])
        self.meshes = self.load_meshes()
        mesh = self.get_forward_hand_mesh(pose, theta)[0]
        mesh.export(os.path.join(BASE_DIR, '../assets/hand_to_mano_frame.obj'))

        self.make_contact_points = make_contact_points
        self.show_mesh = show_mesh
        self.meshes = self.load_meshes()

    def load_meshes(self):
        mesh_dir = os.path.dirname(os.path.realpath(__file__)) + "/../assets/hand_meshes/"
        meshes = {}
        for key, value in self.link_dict.items():
            mesh_filepath = os.path.join(mesh_dir, value)
            link_pre_transform = self.chain.find_link(key).visuals[0].offset
            if self.show_mesh:
                mesh = trimesh.load(mesh_filepath)
                if self.make_contact_points:
                    mesh = trimesh.load(mesh_filepath.replace('assets/hand_meshes/', 'assets/hand_meshes_cvx/'))

                #     # mesh = mesh.convex_hull
                #     mesh_ = mesh.convex_decomposition(
                #         maxConvexHulls=16 if key == 'base_link' else 2,
                #         resolution=800000 if key == 'base_link' else 1000,
                #         minimumVolumePercentErrorAllowed=0.1 if key == 'base_link' else 10,
                #         maxRecursionDepth=10 if key == 'base_link' else 4,
                #         shrinkWrap=True, fillMode='flood', maxNumVerticesPerCH=32,
                #         asyncACD=True, minEdgeLength=2, findBestPlane=False
                #     )
                #     mesh = np.sum(mesh_)
                verts = link_pre_transform.transform_points(torch.FloatTensor(np.array(mesh.vertices)))


                temp = torch.ones(mesh.vertices.shape[0], 1).float()
                vertex_normals = link_pre_transform.transform_normals(torch.FloatTensor(copy.deepcopy(mesh.vertex_normals)))
                meshes[key] = [
                    torch.cat((verts, temp), dim=-1).to(self.device),
                    mesh.faces,
                    torch.cat((vertex_normals, temp), dim=-1).to(self.device).to(torch.float)
                ]
            else:
                vertex_path = mesh_filepath.replace('hand_meshes', 'hand_points').replace('.stl', '.npy').replace('.STL', '.npy').replace('.obj', '.npy')
                assert os.path.exists(vertex_path)
                points_info = np.load(vertex_path)

                link_pre_transform = self.chain.find_link(key).visuals[0].offset
                if self.make_contact_points:
                    idxs = np.arange(len(points_info))
                else:
                    idxs = np.load(os.path.dirname(os.path.realpath(__file__)) + '/../assets/visible_point_indices/{}.npy'.format(key))

                verts = link_pre_transform.transform_points(torch.FloatTensor(points_info[idxs, :3]))
                # print(key, value, verts.shape)
                vertex_normals = link_pre_transform.transform_normals(torch.FloatTensor(points_info[idxs, 3:6]))

                temp = torch.ones(idxs.shape[0], 1)

                meshes[key] = [
                    torch.cat((verts, temp), dim=-1).to(self.device),
                    torch.zeros([0]),  # no real meaning, just for placeholder
                    torch.cat((vertex_normals, temp), dim=-1).to(torch.float).to(self.device)
                ]

        return meshes

    def forward(self, theta):
        """[summary]
        Args:
            pose (Tensor (batch_size x 4 x 4)): The pose of the base link of the hand as a translation matrix.
            theta (Tensor (batch_size x 15)): The seven degrees of freedome of the Barrett hand. The first column specifies the angle between
            fingers F1 and F2,  the second to fourth column specifies the joint angle around the proximal link of each finger while the fifth
            to the last column specifies the joint angle around the distal link for each finger

       """
        # batch_size = pose.shape[0]
        # pose_normal = pose.clone()
        # pose_normal[:, :3, 3] = torch.zeros(3, device=pose.device)
        ret = self.chain.forward_kinematics(theta)
        return ret

    def get_hand_mesh(self, pose, ret):
        bs = pose.shape[0]
        order_keys = [
            'gripper_base',
            'left_finger', 'right_finger'
        ]
        meshes = []
        # for key, item in self.assets.items():
        for key in order_keys:
            rotmat = ret[key].get_matrix()
            # rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform.T, rotmat))
            rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform, rotmat))

            vertices = self.meshes[key][0]
            batch_vertices = torch.matmul(rotmat, vertices.transpose(0, 1)).transpose(1, 2)[..., :3]
            face = self.meshes[key][1]
            sub_meshes = [trimesh.Trimesh(vertices.cpu().numpy(), face) for vertices in batch_vertices]
            # if self.make_contact_points:
            #     tmp_mesh = np.sum(sub_meshes)
            #     tmp_mesh.export('../assets/502_hand_composite/{}.stl'.format(key))
            meshes.append(sub_meshes)

        hand_meshes = []
        for j in range(bs):
            hand = [meshes[i][j] for i in range(len(meshes))]
            hand_mesh = np.sum(hand)
            hand_meshes.append(hand_mesh)
        return hand_meshes

    def get_forward_hand_mesh(self, pose, theta):
        # batch_size = pose.size()[0]
        outputs = self.forward(theta)
        hand_meshes = self.get_hand_mesh(pose, outputs)

        return hand_meshes

    def get_forward_vertices(self, pose, theta):
        # batch_size = pose.size()[0]
        outputs = self.forward(theta)
        verts = []
        verts_normal = []
        order_keys = [
            'gripper_base',
            'left_finger', 'right_finger'
        ]
        # for key, item in self.assets.items():
        for key in order_keys:
            rotmat = outputs[key].get_matrix()
            # rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform.T, rotmat))
            rotmat = torch.matmul(pose, torch.matmul(self.to_mano_transform, rotmat))

            vertices = self.meshes[key][0]
            vertex_normals = self.meshes[key][2]
            batch_vertices = torch.matmul(rotmat, vertices.transpose(0, 1)).transpose(1, 2)[..., :3]
            verts.append(batch_vertices)
            if not os.path.exists('../assets/hand_composite_points'):
                os.makedirs('../assets/hand_composite_points', exist_ok=True)
            if self.make_contact_points:
                np.save('../assets/hand_composite_points/{}.npy'.format(key),
                        batch_vertices.squeeze().cpu().numpy())
            rotmat[:, :3, 3] = 0
            batch_vertex_normals = torch.matmul(rotmat, vertex_normals.transpose(0, 1)).transpose(1, 2)[..., :3]
            verts_normal.append(batch_vertex_normals)

        verts = torch.cat(verts, dim=1).contiguous()
        verts_normal = torch.cat(verts_normal, dim=1).contiguous()
        return verts, verts_normal


def check_assets():
    # ###################################################################################################
    hand = ZYParallelGripperLayer(show_mesh=True, make_contact_points=False, to_mano_frame=False, device=device)
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    theta = np.zeros((1, 2), dtype=np.float32)
    theta = torch.from_numpy(theta).to(device)
    mesh = hand.get_forward_hand_mesh(pose, theta)[0]
    mesh.show()
    # ###################################################################################################

    hand = ZYParallelGripperLayer(show_mesh=True, make_contact_points=False, to_mano_frame=True, device=device)
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    theta = np.zeros((1, 2), dtype=np.float32)
    theta = torch.from_numpy(theta).to(device)
    mesh = hand.get_forward_hand_mesh(pose, theta)[0]
    mesh.show()
    # ###################################################################################################

    hand = ZYParallelGripperLayer(show_mesh=True, make_contact_points=False, to_mano_frame=True, device=device)
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    theta = np.array([GRIPPER_MAX_WIDTH, GRIPPER_MAX_WIDTH], dtype=np.float32)
    theta = torch.from_numpy(theta).to(device)
    mesh = hand.get_forward_hand_mesh(pose, theta)[0]
    mesh.show()
    # ###################################################################################################

    hand = ZYParallelGripperLayer(show_mesh=False, make_contact_points=False, to_mano_frame=False, device=device)
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    theta = np.zeros((1, 2), dtype=np.float32)
    theta = torch.from_numpy(theta).to(device)
    verts, normals = hand.get_forward_vertices(pose, theta)
    pc = trimesh.PointCloud(verts.squeeze().cpu().numpy(), colors=(0, 255, 255))
    mesh = trimesh.load(os.path.join(BASE_DIR, '../assets/hand_all_zero.obj'))
    scene = trimesh.Scene([pc, mesh])
    scene.show()
    # ###################################################################################################

    hand = ZYParallelGripperLayer(show_mesh=False, make_contact_points=False, to_mano_frame=True, device=device)
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    theta = np.zeros((1, 2), dtype=np.float32)
    theta = torch.from_numpy(theta).to(device)
    verts, normals = hand.get_forward_vertices(pose, theta)
    pc = trimesh.PointCloud(verts.squeeze().cpu().numpy(), colors=(0, 255, 255))
    mesh_to_mano = trimesh.load(os.path.join(BASE_DIR, '../assets/hand_to_mano_frame.obj'))
    scene = trimesh.Scene([pc, mesh_to_mano])
    scene.show()
    # ###################################################################################################

    hand = ZYParallelGripperLayer(show_mesh=False, make_contact_points=False, to_mano_frame=True, device=device)
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()
    theta = np.array([GRIPPER_MAX_WIDTH, GRIPPER_MAX_WIDTH], dtype=np.float32)
    theta = torch.from_numpy(theta).to(device)
    verts, normals = hand.get_forward_vertices(pose, theta)
    pc = trimesh.PointCloud(verts.squeeze().cpu().numpy(), colors=(0, 255, 255))
    mesh_to_mano = trimesh.load(os.path.join(BASE_DIR, '../assets/hand_to_mano_frame.obj'))
    scene = trimesh.Scene([pc, mesh_to_mano])
    scene.show()


if __name__ == "__main__":
    device = 'cuda'
    # 0. to_mano_frame=False
    # 1. True False;
    # 2. True True;
    # 3. False True

    # hand = ZYParallelGripperLayer(show_mesh=True, make_contact_points=True, to_mano_frame=False, device=device)
    # exit()

    check_assets()
    exit()

    show_mesh = True
    make_contact_points = False

    export_mesh = False

    to_mano_frame = True
    # if make_contact_points:
    #     from convexhull import sample_points_on_mesh, sample_visible_points
    #     sample_points_on_mesh(src_dir='../assets/hand_meshes')

    hand = ZYParallelGripperLayer(show_mesh=show_mesh, make_contact_points=make_contact_points, to_mano_frame=to_mano_frame, device=device)
    print('Prepare finished here')
    pose = torch.from_numpy(np.identity(4)).to(device).reshape(-1, 4, 4).float()

    # theta = np.zeros((1, 10), dtype=np.float32)
    # theta = np.ones((1, 10), dtype=np.float32) * 1.1
    theta = np.array([
        # root, tip
        0.0255,
        0.0255
    ], dtype=np.float32)
    theta = torch.from_numpy(theta).to(device)

    # mesh version
    if show_mesh:
        mesh = hand.get_forward_hand_mesh(pose, theta)[0]
        mesh.show()
        if export_mesh:
            # if make_contact_points:
            #     mesh.export(os.path.join(BASE_DIR, '../assets/hand.obj'))
            # mesh.export(os.path.join(BASE_DIR, '../assets/hand_all_zero.obj'))
            mesh.export(os.path.join(BASE_DIR, '../assets/hand_to_mano_frame.obj'))
    else:
        if make_contact_points:

            # vertices version
            verts, normals = hand.get_forward_vertices(pose, theta)
            pc = trimesh.PointCloud(verts.squeeze().cpu().numpy(), colors=(0, 255, 255))
            pc.show()
            if make_contact_points:
                sample_visible_points()
        else:
            verts, normals = hand.get_forward_vertices(pose, theta)
            # print(verts.shape)
            pc = trimesh.PointCloud(verts.squeeze().cpu().numpy(), colors=(0, 255, 255))
            # pc.show()
            mesh = trimesh.load(os.path.join(BASE_DIR, '../assets/hand_all_zero.obj'))
            mesh_to_mano = trimesh.load(os.path.join(BASE_DIR, '../assets/hand_to_mano_frame.obj'))
            scene = trimesh.Scene([pc, mesh])
            if to_mano_frame:
                scene = trimesh.Scene([pc, mesh_to_mano])
            # scene.show()


            anchor_layer = SpeedHandAnchor()
            # anchor_layer.pick_points(verts.squeeze().cpu().numpy())
            anchors = anchor_layer(verts)
            pc_anchors = trimesh.PointCloud(anchors.squeeze().cpu().numpy(), colors=(0, 0, 255))
            # pc_anchors.show()
            scene = trimesh.Scene([pc_anchors, mesh])
            if to_mano_frame:
                scene = trimesh.Scene([pc_anchors, mesh_to_mano])
            scene.show()
            # for idx in range(1, 46 + 1):
            #     pc_anchor_1 = trimesh.PointCloud(anchors.squeeze().cpu().numpy()[:idx, :], colors=(0, 0, 255))
            #     pc_anchor_2 = trimesh.PointCloud(anchors.squeeze().cpu().numpy()[idx:, :], colors=(255, 0, 255))
            #     scene = trimesh.Scene([pc_anchor_1, pc_anchor_2])
            #     scene.show()
