import numpy as np
from IPython import embed
from numpy.linalg import norm

def normalize(v):
    return v/(norm(v) + 1e-6)

def get_curve(ps, pe, ds, de, rs, re, scalar=200):
    
    length = np.sqrt(np.sum((ps - pe) ** 2))
    ps_control = ps + ds * 0.1 * length
    pe_control = pe - de * 0.9 * length

    samples = [i/scalar for i in range(scalar+1)]

    points = []
    dirs = []
    radius = []

    for t in samples:
        p = (1-t)**3 * ps + 3 * (1-t)**2*t * ps_control + 3 * (1-t)*t**2 * pe_control + t**3 * pe
        d = ds * (1-t) + de * t
        r = rs * (1-t) + re * t
        points.append(p)
        dirs.append(d)
        radius.append(r)
    
    for i, dir in enumerate(dirs):
        if sum(dir) == 0:
            dirs[i] = dirs[i-1]
            
    points = np.array(points)
    dirs = np.array(dirs)
    radius = np.array(radius)
        
    return points, dirs, radius


def sample_circle(point, dir, r, sample_num=10):

    '''
    input:
    points: 3
    dirs  : 3
    r     : 1

    '''
    point += np.random.rand(3) / 10000
    x_bar = np.cross(point, dir)
    x_bar /= np.sqrt(np.sum(x_bar**2) + 1e-6)

    y_bar = np.cross(dir, x_bar)
    y_bar /= np.sqrt(np.sum(y_bar**2)+ 1e-6)

    theta = np.array([i/sample_num for i in range(sample_num)]) * 2 * np.pi

    sampled_points = point[None, :] + r * (np.cos(theta)[:, None] * x_bar[None, :] + np.sin(theta)[:, None] * y_bar[None, :])
    sampled_points = sampled_points.reshape(-1, 3)

    return sampled_points


def rotation_matrix_from_vectors(v1, v2):
    # Ensure that the vectors are unit vectors
    v1 = v1 / (np.linalg.norm(v1)+1e-6)
    v2 = v2 / (np.linalg.norm(v2)+1e-6)

    # Calculate the rotation axis and angle
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))

    # If the vectors are already parallel, return the identity matrix
    if np.allclose(axis, 0):
        return np.eye(3)

    # Normalize the axis
    axis /= np.linalg.norm(axis)

    # Rodrigues' rotation formula
    k = axis
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return rotation_matrix


def random_sample_circle(point, dir, r, sample_num=20, axis=-1):

    height = np.linalg.norm(dir)
    dir /= height
    
    theta = np.linspace(0, -2*np.pi, sample_num)
    sampled_points = np.array([r/2 * np.cos(theta), np.zeros(theta.shape[0]), r/2 * np.sin(theta)])
    rotation_matrix = rotation_matrix_from_vectors(np.array([0, 1, 0]), dir)
    translation_matrix = np.array(point)[None, :]
        
    return np.dot(rotation_matrix, sampled_points).T + translation_matrix


def gen_tree_mesh(points, dirs, rs, start_idx, sample_num=20):
    
    total_points = []
    total_faces = []
    
    # generate spline for cylinder
    scalar = int(np.ceil(np.sqrt(np.sum((points[0]-points[1])**2)) / 0.005))
    curr_points, curr_dirs, curr_radius = get_curve(points[0], points[1], dirs[0], dirs[1], rs[0], rs[1], scalar=scalar)
    
    
    # sample all nodes for each circle
    for point, dir, r in zip(curr_points, curr_dirs, curr_radius):
        if np.sum(dir) == 0:
            print("dir divided 0")
            embed()
            
        sampled_points = random_sample_circle(point, dir, r, sample_num)
        total_points.append(sampled_points)
    
    # find two layers for mesh generation
    for layer in range(len(total_points)-1):

        node = layer * sample_num + start_idx
        for i in range(sample_num-1):

            # find the start node for the mesh
            total_faces.append([3, node, node + sample_num + 1, node + sample_num])
            total_faces.append([3, node, node + 1, node + sample_num + 1])
            node += 1
        
        # find the last mesh for this layer
        total_faces.append([3, node, node + sample_num + 1 - sample_num, node + sample_num])
        total_faces.append([3, node, node + 1 - sample_num, node + sample_num + 1 - sample_num])
    
    total_points = np.concatenate(total_points, axis=0)

    return np.array(total_points), np.array(total_faces)


def rotation_matrix_from_vectors(v1, v2):
    # Ensure that the vectors are unit vectors
    v1 = v1 / (np.linalg.norm(v1)+1e-6)
    v2 = v2 / (np.linalg.norm(v2)+1e-6)

    # Calculate the rotation axis and angle
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))

    # If the vectors are already parallel, return the identity matrix
    if np.allclose(axis, 0):
        return np.eye(3)

    # Normalize the axis
    axis /= np.linalg.norm(axis)

    # Rodrigues' rotation formula
    k = axis
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return rotation_matrix


def generate_random_vector_with_angle(v, angle_deg):
    """
    Generate a random vector in 3D that forms a specified angle with a given normalized vector.

    Args:
        v (numpy.ndarray): A normalized 3D vector.
        angle_deg (float): Desired angle in degrees.

    Returns:
        numpy.ndarray: A random vector forming the specified angle with v.
    """
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle_deg)

    # Generate a random vector perpendicular to v
    while True:
        random_vec = np.random.randn(3)  # Generate a random vector
        # Compute cross product to find a perpendicular vector
        perp_vec = np.cross(v, random_vec)
        if np.linalg.norm(perp_vec) > 1e-6:  # Ensure it's not zero
            perp_vec = perp_vec / np.linalg.norm(perp_vec)  # Normalize
            break

    # Create the new vector forming the desired angle
    new_vector = np.cos(angle_rad) * v + np.sin(angle_rad) * perp_vec
    return new_vector
