from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import trimesh
# import pybullet as op1
# from pybullet_utils import bullet_client
import multiprocessing as mp
import shutil
import coacd
import open3d as o3d
import numpy as np


def normalize_obj(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.split('.')[1] == 'obj':
                file_path = os.path.join(root, file_name)
                mesh = trimesh.load_mesh(file_path)
                file_path_obj = file_path.split('.')[0] + '.obj'
                # trimesh.smoothing.filter_humphrey(mesh, alpha=0.1, beta=0.5, iterations=10, laplacian_operator=None)

                mesh.vertices -= mesh.center_mass
                # mesh.show()
                mesh.export(file_path_obj)
                if not mesh.is_watertight:
                    print(file_name, 'not watertight')


def make_obj_dir(root_dir, dst_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if not file_name.split('.')[1] == 'obj':
                continue
            if '_simplified' in file_name:
                continue
            print(file_name)
            filepath = os.path.abspath(os.path.join(root, file_name))
            folder_name = file_name.split('/')[-1].split('.')[0]
            folder_path = os.path.join(dst_dir, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)

            shutil.copy(filepath, folder_path)


def remove_stl(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.split('.')[1] == 'stl':
                file_path = os.path.join(root, file_name)
                os.remove(file_path)


def obj2stl(root_dir):
    # BHAM_obj_path = '/home/v-wewei/hand/BHAM_vhacd_stl'
    # if not os.path.exists(BHAM_obj_path):
    #     os.mkdir(BHAM_obj_path)

    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.split('.')[1] == 'obj' and '_vhacd' not in file_name and '_vox' not in file_name:
                file_path = os.path.join(root, file_name)
                mesh = trimesh.load_mesh(file_path)
                file_path_obj = file_path.split('.')[0] + '.stl'
                mesh.export(file_path_obj)
                print(file_path_obj, 'fininsh')


def remove_piece_stl(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.split('.')[1] == 'stl' and '_cvx_' in file_name:
                file_path = os.path.join(root, file_name)
                os.remove(file_path)


def remove_vhacd(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.split('.')[1] == 'obj' and '_vhacd' in file_name:
                file_path = os.path.join(root, file_name)
                os.remove(file_path)


def vhacd(name_in,name_out,name_log):
    #pb_client = bullet_client.BulletClient(op1.DIRECT)
    #pb_client.vhacd(name_in, name_out, name_log, concavity=0.0001, gamma=0.0001, maxNumVerticesPerCH=64, resolution=500000)
    os.system('testVHACD --input {} --output {} --maxhulls 24 --concavity 0.0001 --gamma 0.0001 --maxNumVerticesPerCH 48  --resolution 5000000 --log log.txt'.format(name_in, name_out))
    # os.system(
    #     'TestVHACD {}  -h 32  -v 64 -r 1000000'.format(
    #         name_in))


def write_convex_obj_file(path_to_obj_file, parts):
    objFile = open(path_to_obj_file, 'w')
    idx = 0
    bais = 0
    for verts, faces in parts:
        objFile.write("o convex_{} \n".format(idx))
        for vert in verts:
            objFile.write("v ")
            objFile.write(str(vert[0]))
            objFile.write(" ")
            objFile.write(str(vert[1]))
            objFile.write(" ")
            objFile.write(str(vert[2]))
            objFile.write("\n")
        for face in faces:
            objFile.write("f ")
            objFile.write(str(face[0] + 1+bais))
            objFile.write(" ")
            objFile.write(str(face[1] + 1+bais))
            objFile.write(" ")
            objFile.write(str(face[2] + 1+bais))
            objFile.write("\n")
        bais += len(verts)
        idx += 1
    objFile.close()


def coacd_decomposition(name_in, name_out, name_log):
    mesh = trimesh.load(name_in, force="mesh")
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    # parts = coacd.run_coacd(coacd_mesh, threshold=0.05, max_convex_hull=12)
    parts = coacd.run_coacd(coacd_mesh, threshold=0.03)
    for idx, part in enumerate(parts):
        name = name_out.replace('_vhacd.obj', '_cvx_{}.stl'.format(idx))
        verts, faces = part
        tmp_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        tmp_mesh.export(name)

    # parts = coacd.run_coacd(coacd_mesh, threshold=0.03)
    write_convex_obj_file(path_to_obj_file=name_out, parts=parts)


def vhacd_to_piece(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.split('.')[1] == 'obj' and '_vhacd.obj' in file_name and '_vox_' not in file_name:
                file_path = os.path.join(root, file_name)
                meshes = trimesh.load_mesh(file_path)
                mesh_list = meshes.split()
                for i, mesh in enumerate(mesh_list):
                    new_file_path = file_path.replace('vhacd.obj', 'cvx_{}.stl'.format(i))
                    mesh.export(new_file_path)


def create_xml(file_dir_name, root_dir):
    # create the file structure
    data = ET.Element('mujoco')
    data.set('model', 'OBJ')
    compiler = ET.SubElement(data, 'compiler')
    size = ET.SubElement(data, 'size')
    compiler.set('angle', 'radian')
    size.set('njmax', '500')
    size.set('nconmax', '100')
    item_asset = ET.SubElement(data,'asset')

    item_wordbody = ET.SubElement(data,'worldbody')
    item_empty_body = ET.SubElement(item_wordbody, 'body')
    if not static:
        item_joint_tx = ET.SubElement(item_empty_body, 'joint')
        item_joint_tx.set('name', 'free_joint')
        item_joint_tx.set('type', 'free')
        item_joint_tx.set('damping', '0.0')
    item_body = ET.SubElement(item_empty_body, 'body')
    # item_body.set('name', file_dir_name)
    item_body.set('name', 'object')
    item_body.set('pos','0 0 0')
    item_body.set('euler','0 0 0')

    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            # create visual attribute
            if '_cvx_' not in file_name and '_vhacd' not in file_name and file_name.split('.')[-1] == 'obj':
                file_name_ = file_name.split('.')[0]
                file_path = os.path.abspath(os.path.join(root, file_name))

                item_mesh = ET.SubElement(item_asset, 'mesh')
                item_mesh.set('name', file_name_)
                item_mesh.set('file', file_path)

                item_geom = ET.SubElement(item_body, 'geom')
                item_geom.set('type', 'mesh')
                item_geom.set('density', '0')
                item_geom.set('mesh', file_name_)
                item_geom.set('name', file_name_)
                item_geom.set('contype','0')
                item_geom.set('conaffinity','0')
                item_geom.set('group','1')

            # create collision attribute
            if '.stl' in file_name and 'cvx' in file_name and file_name.split('.')[0].split('_')[-2] == 'cvx':
                file_name_ = file_name.split('.')[0]
                file_path = os.path.abspath(os.path.join(root, file_name))
                item_mesh = ET.SubElement(item_asset, 'mesh')
                item_mesh.set('name', file_name_)

                item_mesh.set('file', file_path)

                item_geom = ET.SubElement(item_body, 'geom')
                item_geom.set('type','mesh')
                item_geom.set('density','1500')
                item_geom.set('mesh',file_name_)
                item_geom.set('name', file_name_)
                item_geom.set('group', '0')
                item_geom.set('condim', '4')
                item_geom.set('friction', '0.2 0.005 0.0001')
                # item_geom.set('solref', "0.02 1.0")
                # item_geom.set('friction', '10')

    et = ET.ElementTree(data)
    if static:
        dst_dir = os.path.join(root_dir, '../obj_xml_static')
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        fname = os.path.join(root_dir, '../obj_xml_static/{}.xml'.format(file_dir_name))
    else:
        dst_dir = os.path.join(root_dir, '../obj_xml')
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        fname = os.path.join(root_dir, '../obj_xml/{}.xml'.format(file_dir_name))
    et.write(fname, encoding='utf-8', xml_declaration=True)
    x = minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent='  '))


def sample_points(mesh_filepath, ply_savepath, target_num=5000, use_box_sample=True, oriented_box=False):
    if not use_box_sample:
        mesh = o3d.io.read_triangle_mesh(mesh_filepath)
    else:
        tmp_mesh = trimesh.load_mesh(mesh_filepath)
        if oriented_box:
            box_mesh = tmp_mesh.bounding_box_oriented
        else:
            box_mesh = tmp_mesh.bounding_box
        mesh = box_mesh.as_open3d
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])

    pointcloud = mesh.sample_points_poisson_disk(number_of_points=target_num, use_triangle_normal=True)
    pc = np.array(pointcloud.points)
    normal = np.array(pointcloud.normals)

    pointcloud.points = o3d.utility.Vector3dVector(pc)
    pointcloud.normals = o3d.utility.Vector3dVector(normal)
    o3d.io.write_point_cloud(ply_savepath, pointcloud, write_ascii=True)


if __name__ == '__main__':
    src_dir = '../asset/objs'
    dst_dir = '../asset/mujoco_asset'

    # step1: object to folder
    make_obj_dir(src_dir, dst_dir)

    # step2: Mesh Decomposition for collision shapes used by mujoco
    static = False
    to_vhacd = True
    if to_vhacd:
        pool = mp.Pool(int(mp.cpu_count() / 2))

        for root, dirs, files in os.walk(dst_dir):
            for file_name in files:
                if file_name.endswith('.obj') and '_vox_' not in file_name and "_vhacd" not in file_name:
                    name_in = os.path.join(root, file_name)
                    name_out = name_in.replace('.obj', '_vhacd.obj')
                    name_log = "log.txt"
                    if not os.path.exists(name_out):
                        pool.apply_async(coacd_decomposition, args=(name_in, name_out, name_log,))
        pool.close()
        pool.join()

    # step3: create xml file for mujoco simulation
    for root, dirs, files in os.walk(dst_dir):
        for dir_name in dirs:
            src = os.path.join(root, dir_name)
            create_xml(dir_name, src)
            print('{} is ok'.format(src))

    # step4: sample grasp approach point
    sample_graspable_points = True
    if sample_graspable_points:
        target_point_num = 1024
        for root, dirs, files in os.walk(dst_dir):
            for filename in files:
                if filename.endswith('.obj') and '_vhacd' not in filename:
                    filepath = os.path.join(root, filename)
                    ply_filepath = filepath.replace('.obj', '_{}.ply'.format(target_point_num))
                    sample_points(filepath, ply_filepath, target_num=target_point_num, use_box_sample=True)
    

