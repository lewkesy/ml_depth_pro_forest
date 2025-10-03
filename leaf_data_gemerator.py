import trimesh
import numpy as np
import os
from collections import OrderedDict
from IPython import embed
from utils.geometry import normalize, gen_tree_mesh, rotation_matrix_from_vectors, generate_random_vector_with_angle
from utils.visualize import save_mesh


def get_mesh(file_path): 
    
    mesh = trimesh.load(file_path, force='mesh') 
    vertices = mesh.vertices.tolist()
    
    if ".off" in file_path:  # ModelNet dataset
       mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]] 
       rotation_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
       mesh.apply_transform(rotation_matrix) 
        # Extract vertices and faces from the rotated mesh
       vertices = mesh.vertices.tolist()
            
    faces = mesh.faces.tolist()
    
    centered_vertices = vertices - np.mean(vertices, axis=0)  
    max_abs = np.max(np.abs(centered_vertices))
    vertices = centered_vertices / (max_abs / 0.95)     # Limit vertices to [-0.95, 0.95]
    
    min_y = np.min(vertices[:, 1]) 
    difference = -0.95 - min_y 
    vertices[:, 1] += difference
    
    def sort_vertices(vertex):
        return vertex[1], vertex[2], vertex[0]   
 
    seen = OrderedDict()
    for point in vertices: 
      key = tuple(point)
      if key not in seen:
        seen[key] = point
        
    unique_vertices =  list(seen.values()) 
    sorted_vertices = sorted(unique_vertices, key=sort_vertices)
      
    vertices_as_tuples = [tuple(v) for v in vertices]
    sorted_vertices_as_tuples = [tuple(v) for v in sorted_vertices]

    vertex_map = {old_index: new_index for old_index, vertex_tuple in enumerate(vertices_as_tuples) for new_index, sorted_vertex_tuple in enumerate(sorted_vertices_as_tuples) if vertex_tuple == sorted_vertex_tuple} 
    reindexed_faces = [[vertex_map[face[0]], vertex_map[face[1]], vertex_map[face[2]]] for face in faces] 
    sorted_faces = [sorted(sub_arr) for sub_arr in reindexed_faces]   
    
    vertices = np.array(sorted_vertices)
    faces = np.array(sorted_faces)

    mesh = []
    for face in faces:
        mesh.append(vertices[face].tolist())
    mesh = np.array(mesh)
    mesh = np.array(sorted(mesh, key=lambda x:np.sum(x[:, 1])))
    return vertices, faces, mesh


def build_connectivity(mesh):
    mesh1 = mesh[:, None, :, None, :]
    mesh2 = mesh[None, :, None, :, :]
    mesh_dist = np.sqrt(np.sum((mesh1 - mesh2) ** 2, axis=-1))

    # only consider meshes with only one common edge
    connectivity = np.sum(mesh_dist <= 1e-5, axis=(-1, -2)) >= 2
    print("Connectivity build finished")
    return connectivity


def build_graph(graph, mesh, min_radius=0.0025, radius_offset_factor=0.005):
    tree_graph = []
    visited = set([-1]) # visited triangles in mesh
    queue = [(0,-1)] # current triangle in mesh, previous node in tree_graph
    
    while queue:
        curr_triangle_idx, parent_node_idx = queue.pop(0)
        parent_radius = tree_graph[parent_node_idx]['radius'] if parent_node_idx != -1 else 1
        curr_vertices = mesh[curr_triangle_idx]

        if curr_triangle_idx in visited:
            continue
        
        # find potential connectivity
        connected_triangle_mesh_candidate = np.where(graph[curr_triangle_idx]==True)[0]
        connected_triangle_mesh_idx = [idx for idx in connected_triangle_mesh_candidate if idx not in visited]
        connected_triangle_mesh_idx_initial = [idx for idx in connected_triangle_mesh_candidate]
        # find two vtx with the same height
        y_sum = np.sum(abs(curr_vertices[:, 1][None, :] - curr_vertices[:, 1][:, None]), axis=1)/2
        idx1, idx2, idx3 = np.argsort(y_sum)

        
        # find the center point of two vtx as the node position
        pos = (curr_vertices[idx1] + curr_vertices[idx2]) / 2
        radius = abs(curr_vertices[idx1][2] - curr_vertices[idx2][2]) - radius_offset_factor
        radius = radius if radius < parent_radius else parent_radius
        
        if radius < 0:
            continue
        else:
            radius = max(radius, min_radius)
        
        # if radius == 0:
            # fix issue when two vertices are the same
            # curr_vertices[idx2, 1] += parent_radius
            # pos = (curr_vertices[idx1] + curr_vertices[idx2]) / 2
            # radius = abs(curr_vertices[idx1][2] - curr_vertices[idx2][2]) - radius_offset_factor
            # radius = min(radius, parent_radius)
            # radius = max(radius, min_radius)
            
        # if triangle does not share the same pos with parent triangle
        # if len(connected_triangle_mesh_idx) == 0, it is the last triangle with upper triangle
        if parent_node_idx == -1 or np.sqrt(np.sum((pos - tree_graph[parent_node_idx]['pos'])**2)) > 1e-5 or len(connected_triangle_mesh_idx_initial) == 1:
            
            curr_node_idx = len(tree_graph)
            node = {
                'pos': pos,
                'radius': float(radius),
                'dir': None,
                'children': []
            }
            
                
            if parent_node_idx != -1:
                if len(connected_triangle_mesh_idx_initial) != 1:
                    tree_graph[parent_node_idx]['children'].append(node)
                    node['dir'] = normalize(pos - tree_graph[parent_node_idx]['pos'])
                    
                    if np.sum((pos - tree_graph[parent_node_idx]['pos'])** 2) < 1e-5:
                        node['dir'] = tree_graph[parent_node_idx]['dir'].copy()

                else:
                    # fix the single triangle on the twigs
                    tree_graph[parent_node_idx]['children'].append(node)
                    node['pos'] = curr_vertices[idx3]
                    node['dir'] = normalize(node['pos'] - tree_graph[parent_node_idx]['pos'])
            else:
                node['dir'] = np.array([0, 1, 0])
            
            
                
            for connect in connected_triangle_mesh_idx:
                if connect not in visited:
                    queue.append((connect, curr_node_idx))
                    
            tree_graph.append(node)
            visited.add(curr_triangle_idx)
        
        # if triangle share the same pos, meaning this triangle will not be interpolated as a node
        else:
            for connect in connected_triangle_mesh_idx:
                if connect not in visited:
                    queue.append((connect, parent_node_idx))
            visited.add(curr_triangle_idx)
    
    print("Tree graph generated")
    return tree_graph


def convert_tree_graph(filepath, tree_graph, debug=None):
    queue = [tree_graph[0]]
    vertices, faces = [], []
    
    cnt = 0
    while queue:

        cnt += 1
        curr_node = queue.pop(0)
        children = curr_node['children']
        curr_pos, curr_dir, curr_radius = curr_node['pos'], curr_node['dir'], curr_node['radius']
        
        if debug and cnt > debug:
            print(curr_node)
            break        
        
        for child in children:
            child_pos, child_dir, child_radius = child['pos'], child['dir'], child['radius']
            child_vertices, child_mesh = gen_tree_mesh(
                [curr_pos, child_pos],
                [curr_dir, child_dir],
                [curr_radius, child_radius],
                start_idx=0, 
                sample_num=20
            )
            
            child_mesh[:, 1:] += len(vertices)
            faces += child_mesh.tolist()
            vertices += child_vertices.tolist()
            
        # add children to the queue
        for child in children:
            queue.append(child)
    
    # embed()
    save_mesh(filepath, np.array(vertices), np.array(faces))
    return


def interpolate(start, end, factor=5):
    return [start * (1 - i/factor) + end * i/factor for i in range(factor+1)]

def generate_foliage_mesh(tree_graph, sub_tree_root, foliage_radius=0, quad_size=0.012, degree=30, min_radius=0.005):
    
    min_radius=min_radius
    visited_node = []
    queue = [sub_tree_root]
    foliage_total_points = []
    foliage_total_faces = []
    
    foliage_per_node = 1
    vertice_cnt = 0
    
    while queue:
        node = queue.pop(0)
        visited_node.append(node)
        for child in node['children']:
            
            queue.append(child)
            
            interpolation_factor = int(2 / 0.06 * np.sqrt(((node['pos'] - child['pos'])**2).sum())) + 1

            points = interpolate(node['pos'], child['pos'], interpolation_factor)
            rs = interpolate(node['radius'], max(min(child['radius'], node['radius']/2), 0), interpolation_factor)
            dirs = interpolate(node['dir'], child['dir'], interpolation_factor)
            
            dist = np.sqrt(((points[1] - points[0])**2).sum())
            
            for point, dir, r in zip(points, dirs, rs):
                
                if r == 0:
                    # print(rs)
                    r = min_radius
                    # continue
                
                for _ in range(foliage_per_node):
                    # quad = np.array([[-0.5,0,-0.5], 
                    #         [-0.5,0,0.5], 
                    #         [0.5,0,0.5], 
                    #         [0.5,0,-0.5]])
                    quad = np.array([[0,0,-0.2], 
                            [0,0,0.2], 
                            [1,0,0.2], 
                            [1,0,-0.2]])

                    leaf_axis_dir = np.array([1, 0, 0])
                    
                    ratio = foliage_radius if foliage_radius != 0 else 1
                    
                    seed = (np.random.rand(3,)*2-1)
                    seed[1] *= 0.1
                    position = seed*r*ratio + point
                    lookat_position = position+dir*dist*0.75
                    plane_rotation_matrix = rotation_matrix_from_vectors(np.array([0,1,0]), lookat_position-position)
                    target_leaf_axis_dir = generate_random_vector_with_angle(dir, degree)
                    leaf_rotation_matrix = rotation_matrix_from_vectors(leaf_axis_dir, target_leaf_axis_dir)
                    
                    # rotation_matrix = rotation_matrix_from_vectors(np.array([0,1,0]), lookat_position-position)
                    # quad = (rotation_matrix@plane_rotation_matrix@quad.T).T * quad_size + position
                    quad = (leaf_rotation_matrix@quad.T).T * quad_size + position
                    
                    # embed()
                    
                    foliage_total_points.append(quad)
                    foliage_total_faces.append([vertice_cnt, vertice_cnt+1, vertice_cnt+2])
                    foliage_total_faces.append([vertice_cnt+2, vertice_cnt+3, vertice_cnt])
                    vertice_cnt += 4
    
    if len(foliage_total_points) != 0:
        foliage_total_points = np.concatenate(foliage_total_points, axis=0)
    else:
        foliage_total_points = None
        
    return foliage_total_points, foliage_total_faces, visited_node


def find_sub_tree_root(graph, ratio=0.01):
    candidates = []
    queue = [graph[0]]
    

    while queue:
        curr_node = queue.pop(0)

        if curr_node['radius'] < ratio:
            candidates.append(curr_node)
        else:
            for child in curr_node['children']:
                queue.append(child)
        
    return candidates


def generate_paper_cut_mesh(tree_graph, sub_tree_root):
    queue = [tuple([sub_tree_root, -1])]
    connectivity_list = []
    node_list = []
    radius_list = []
    dir_list = []
    
    while queue:
        node, previous = queue.pop(0)

        if previous != -1:
            connectivity_list.append([previous, len(node_list)])
            dir_list.append(normalize(node['pos'] - node_list[previous]))
            
        node_list.append(node['pos'])
        radius_list.append(node['radius'])       
        
        for child in node['children']:
            queue.append(tuple([child, len(node_list)-1]))
            
    connectivity_list = np.array(connectivity_list)
    node_list = np.array(node_list)
    radius_list = np.array(radius_list)
    dir_list = np.array(dir_list)

    offset = np.ones_like(node_list) * radius_list[:, None]
    offset[:, :2] *= 0
    
    duplicated_node_list = np.concatenate([node_list, node_list+offset], axis=-1).reshape(-1, 3)
    
    faces = []
    for edge in connectivity_list:
        s, e = edge.tolist()
        faces.append([2*s+1, 2*s, e*2])
        faces.append([2*e, 2*e+1, 2*s+1])
    
    return duplicated_node_list, faces


def generate_foliage_with_paper_cut(filename, tree_graph, sub_trees, root=".", min_radius=0.001):
    
    scale_factor = 8
    for i, sub_tree_root in enumerate(sub_trees):
        
        # for one single sub_tree
        
        tree_vertices, tree_faces = generate_paper_cut_mesh(tree_graph, sub_tree_root) #numpy, list
        foliage_vertices, foliage_faces, visited_node = generate_foliage_mesh(tree_graph, sub_tree_root, foliage_radius=3, quad_size=0.062, degree=45, min_radius=min_radius)
        
        # generate paper_cut_model
        if tree_vertices is not None:
            mesh = trimesh.Trimesh(vertices=tree_vertices.tolist(),
                            faces=tree_faces)
            mesh.export(os.path.join(root, "leaves", filename.split('.')[0]+'_sub_tree_%d.obj'%i))
            
        # # generate stretched paper_cut_model
        # stretched_tree_vertices = tree_vertices.copy()
        # stretched_tree_vertices[:, 2] *= scale_factor
        # mesh = trimesh.Trimesh(vertices=stretched_tree_vertices.tolist(),
        #                 faces=tree_faces)
        # mesh.export(os.path.join(root, "leaves", filename.split('.')[0]+'_stretched_sub_tree_%d.obj'%i))
        
        # generate foliage
        if foliage_vertices is not None:
            mesh = trimesh.Trimesh(vertices=foliage_vertices.tolist(),
                            faces=foliage_faces)
            mesh.export(os.path.join(root, "leaves", filename.split('.')[0]+'_foliage_%d.obj'%i))
        else:
            print(filename.split('.')[0]+'_foliage_%d.obj'%i + " is not generated")
            
        # # generate stretched foliage
        # stretched_foliage_vertices = foliage_vertices.copy()
        # stretched_foliage_vertices[:, 2] *= scale_factor
        # mesh = trimesh.Trimesh(vertices=stretched_foliage_vertices.tolist(),
        #                 faces=foliage_faces)
        # mesh.export(os.path.join(root, "leaves", filename.split('.')[0]+'_stretched_foliage_%d.obj'%i))


def interpolate_tree_graph(tree_graph, min_radius=0.005):
    
    queue = [tree_graph[0]]
    while queue:
        node = queue.pop(0)
        # original_child = node['children'].copy()
        
        for child in node['children']:
            
            interpolation_factor = int(10 / 0.06 * np.sqrt(((node['pos'] - child['pos'])**2).sum())) + 1
            # embed()
            points = interpolate(node['pos'], child['pos'], interpolation_factor)
            rs = interpolate(node['radius'], max(min(child['radius'], node['radius']), min_radius), interpolation_factor)
            dirs = interpolate(node['dir'], child['dir'], interpolation_factor)
            # embed()
            curr_parent = node
            curr_parent['children']=[]
            for point, dir, r in zip(points[1:-1], dirs[1:-1], rs[1:-1]):
                
                r = r if r > min_radius else min_radius
                curr_node = {
                    'pos': point,
                    'radius': float(r),
                    'dir': dir,
                    'children': []
                }
                curr_parent['children'].append(curr_node)
                curr_parent = curr_node
            curr_parent['children'].append(child)
            
        
# filepath = '/data/zhou1178/meshGPT/paper_cut/TreeStructor_1.obj'
filename = 'Germany_Fir_143_0_single'
filepath = './%s.obj'%filename
result_filepath = os.path.join("leaves", "%s_interpolate.ply"%filename)
min_radius = 0.0015

os.system("rm -r ./leaves/*")

vertices, faces, mesh = get_mesh(filepath)
vis_mesh = trimesh.Trimesh(vertices=vertices.tolist(),
                faces=faces.tolist())
vis_mesh.export(os.path.join('leaves', filename+'_vis.obj'))

graph = build_connectivity(mesh)
tree_graph = build_graph(graph, mesh, min_radius=min_radius)
interpolate_tree_graph(tree_graph, min_radius=min_radius)
convert_tree_graph(result_filepath, tree_graph, debug=None)

sub_tree_list = find_sub_tree_root(tree_graph, ratio=0.015)
generate_foliage_with_paper_cut(filename, tree_graph, sub_tree_list, min_radius=min_radius)