import trimesh
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial import ConvexHull, distance
import cvxopt as cvx
cvx.solvers.options['show_progress'] = False


def graspit_measure_hard(force_torque):
    force_torque_array = np.asarray(force_torque)
    # print('force torque is :', force_torque_array)

    forces = force_torque_array[..., :3]
    torques = force_torque_array[..., 3:]

    G = grasp_matrix_new(np.array(forces).transpose(), np.array(torques).transpose())

    # debug = True
    # if debug:
    #     fig = plt.figure()
    #     torques = G[3:, :].T
    #     ax = Axes3D(fig)
    #     ax.scatter(torques[:, 0], torques[:, 1], torques[:, 2], c='b', s=50)
    #     ax.scatter(0, 0, 0, c='k', s=80)
    #     ax.set_xlim3d(-1.5, 1.5)
    #     ax.set_ylim3d(-1.5, 1.5)
    #     ax.set_zlim3d(-1.5, 1.5)
    #     ax.set_xlabel('tx')
    #     ax.set_ylabel('ty')
    #     ax.set_zlabel('tz')
    #     plt.show()


    measure = min_norm_vector_in_facet(G)[0]
    return measure


def grasp_matrix_new(forces, torques):
    num_forces = forces.shape[1]
    num_torques = torques.shape[1]
    if num_forces != num_torques:
        raise ValueError('Need same number of forces and torques')

    num_cols = num_forces

    G = np.zeros([6, num_cols])
    for i in range(num_forces):
        G[:3, i] = forces[:, i]
        G[3:, i] = torques[:, i]   # ZEROS

    return G

def graspit_measure(forces, torques, normals):
    G = grasp_matrix(np.array(forces).transpose(), np.array(torques).transpose(), np.array(normals).transpose())
    measure = min_norm_vector_in_facet(G)[0]
    return measure


def grasp_matrix(forces, torques, normals, soft_fingers=False,
                 finger_radius=0.005, params=None):
    if params is not None and 'finger_radius' in params.keys():
        finger_radius = params.finger_radius
    num_forces = forces.shape[1]
    num_torques = torques.shape[1]
    if num_forces != num_torques:
        raise ValueError('Need same number of forces and torques')

    num_cols = num_forces
    if soft_fingers:
        num_normals = 2
        if normals.ndim > 1:
            num_normals = 2*normals.shape[1]
        num_cols = num_cols + num_normals

    torque_scaling = 1
    G = np.zeros([6, num_cols])
    for i in range(num_forces):
        G[:3,i] = forces[:,i]
        G[3:,i] = torque_scaling * torques[:,i]

    if soft_fingers:
        torsion = np.pi * finger_radius**2 * params.friction_coef * normals * params.torque_scaling
        pos_normal_i = -num_normals
        neg_normal_i = -num_normals + num_normals / 2
        G[3:,pos_normal_i:neg_normal_i] = torsion
        G[3:,neg_normal_i:] = -torsion

    return G


def min_norm_vector_in_facet(facet, wrench_regularizer=1e-10):
    """ Finds the minimum norm point in the convex hull of a given facet (aka simplex) by solving a QP.
    Parameters
    ----------
    facet : 6xN :obj:`numpy.ndarray`
        vectors forming the facet
    wrench_regularizer : float
        small float to make quadratic program positive semidefinite
    Returns
    -------
    float
        minimum norm of any point in the convex hull of the facet
    Nx1 :obj:`numpy.ndarray`
        vector of coefficients that achieves the minimum
    """
    dim = facet.shape[1] # num vertices in facet

    # create alpha weights for vertices of facet
    G = facet.T.dot(facet)
    grasp_matrix = G + wrench_regularizer * np.eye(G.shape[0])

    # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h, Ax = b
    P = cvx.matrix(2 * grasp_matrix)   # quadratic cost for Euclidean dist
    q = cvx.matrix(np.zeros((dim, 1)))
    G = cvx.matrix(-np.eye(dim))       # greater than zero constraint
    h = cvx.matrix(np.zeros((dim, 1)))
    A = cvx.matrix(np.ones((1, dim)))  # sum constraint to enforce convex
    b = cvx.matrix(np.ones(1))         # combinations of vertices

    sol = cvx.solvers.qp(P, q, G, h, A, b)
    v = np.array(sol['x'])
    min_norm = np.sqrt(sol['primal objective'])

    return abs(min_norm), v


def get_new_normals(force_vector, normal_force, sides, radius):
    return_vectors = []
    # get arbitrary vector to get cross product which should be orthogonal to both
    vector_to_cross = np.array((force_vector[0] + 1, force_vector[1] + 2, force_vector[2] + 3))
    orthg = np.cross(force_vector, vector_to_cross)
    orthg_vector = (orthg / np.linalg.norm(orthg)) * radius
    rot_angle = (2 * np.pi) / sides
    split_force = normal_force / sides

    for side_num in range(0, sides):
        rotated_orthg = Quaternion(axis=force_vector, angle=(rot_angle * side_num)).rotate(orthg_vector)

        new_vect = force_vector + np.array(rotated_orthg)
        norm_vect = (new_vect / np.linalg.norm(new_vect)) * split_force
        return_vectors.append(norm_vect)

    return return_vectors


def gws_pyramid_extension(object_mesh, points, forces, pyramid_sides=6, pyramid_radius=.4):
    max_radius = object_mesh.bounding_box_oriented.primitive.extents
    obj_pos = object_mesh.center_mass
    pc = trimesh.PointCloud(obj_pos.reshape(-1, 3), colors=(0, 255, 255))

    force_torque = []
    # closest_points, _, triangle_id = trimesh.proximity.closest_point(object_mesh, points)
    # contact_normals = -object_mesh.face_normals[triangle_id]
    # contact_points = np.concatenate([np.asarray(closest_points), np.asarray(contact_normals)], axis=1)
    contact_points = np.copy(points)
    for point, force in zip(contact_points, forces):
        contact_pos = point[:3]
        normal_vector_on_obj = point[3:]
        normal_force_on_obj = 1  # force
        force_vector = np.array(normal_vector_on_obj) * normal_force_on_obj
        if np.linalg.norm(force_vector) > 0:
            new_vectors = get_new_normals(force_vector, normal_force_on_obj, pyramid_sides, pyramid_radius)
            # print(len(new_vectors))
            # print(new_vectors, force_vector)
            # ray_origin = contact_pos
            # ray_direction = np.concatenate((np.asarray(new_vectors), force_vector.reshape(-1, 3)/8))
            #
            # ray_vis = trimesh.load_path(np.hstack((ray_origin + ray_direction*0, ray_origin + ray_direction)).reshape(-1, 2, 3))
            # scene = trimesh.Scene([object_mesh, ray_vis, pc])
            # scene.show()

            radius_to_contact = np.array(contact_pos) - np.array(obj_pos)

            for pyramid_vector in new_vectors:
                torque_numerator = np.cross(radius_to_contact, pyramid_vector)
                torque_vector = torque_numerator / max_radius
                force_torque.append(np.concatenate([pyramid_vector, torque_vector]))
    return force_torque


def eplison(force_torque):
    """
    get qhull of the 6 dim vectors [fx, fy, fz, tx, ty, tz] created by gws (from contact points)
    get the distance from centroid of the hull to the closest vertex
    """
    # try:
    #     hull = ConvexHull(points=force_torque)
    #     centroid = []
    #     for dim in range(0, 6):
    #         centroid.append(np.mean(hull.points[hull.vertices, dim]))
    #     shortest_distance = 500000000
    #     closest_point = None
    #     for point in force_torque:
    #         point_dist = distance.euclidean(centroid, point)
    #         if point_dist < shortest_distance:
    #             shortest_distance = point_dist
    #             qq = point
    # except:
    #     shortest_distance = 0.0
    hull = ConvexHull(points=force_torque, qhull_options='QJ Pp')
    centroid = []
    for dim in range(0, 6):
        centroid.append(np.mean(hull.points[hull.vertices, dim]))
    shortest_distance = 500000000
    closest_point = None
    for point in force_torque:
        point_dist = distance.euclidean(centroid, point)
        if point_dist < shortest_distance:
            shortest_distance = point_dist
            closest_point = point

    return shortest_distance
