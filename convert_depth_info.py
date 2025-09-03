import os
import sys
import argparse
from IPython import embed
from tqdm import tqdm
import numpy as np
import cv2
import trimesh
import multiprocessing


src = '/data/zhou1178/MeshGPT_TreeStructor_validation/rgb'
depth_src = '/data/zhou1178/MeshGPT_TreeStructor_validation/depth'
dst = '/data/zhou1178/MeshGPT_TreeStructor_validation/depth_aug_npz'
obj_src = '/data/zhou1178/MeshGPT_TreeStructor_validation/meshes'


os.makedirs(dst, exist_ok=True)

def save_ply(fn, xyz, color=None):

    with open(fn, 'w') as f:
        pn = len(xyz)
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % (pn))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for i in range(pn):
            if color is None:
                f.write('%.6f %.6f %.6f\n' % (xyz[i][0], xyz[i][1], xyz[i][2]))
                
                
def generate_depth_data(filenames, dst):
    for filename in tqdm(filenames):
        filepath = os.path.join(src, filename)
        os.system("depth-pro-run -i %s -o %s"%(filepath, dst))


def convert_depth_info(filenames):

    heights = []
    max_height = -1
    for filename in tqdm(filenames):
        
        prefix = filename.split('.')[0]
        if filename.endswith('jpg'):
            continue
        
        depth = np.load(os.path.join(depth_src, prefix+'.npz'))['depth']
        depth = depth.max() - depth
        
        rgb = cv2.imread(os.path.join(src, prefix+'_0.png'))
        rgb = cv2.resize(rgb, (256, 256))
        
        mask = 1 - ((rgb[:,:,0]==255) * (rgb[:,:,1]==255) * (rgb[:,:,2]==255))
        depth = depth * mask
        
        mesh = trimesh.load(os.path.join(obj_src, prefix+".obj"), force='mesh')
        vertices = mesh.vertices
        depth_max = vertices[:, 1].max()
        
        depth = depth / depth.max() * depth_max
        depth = depth * mask
        
        max_height = max(max_height, depth_max)
        
        np.savez(os.path.join(dst, prefix+'.npz'), **{'depth':depth})
        heights.append(depth_max)
        
        pointcloud_vis = []
        for h in range(depth.shape[0]):
            for w in range(depth.shape[1]):
                if mask[h][w]:
                    pointcloud_vis.append([h/256, w/256, depth[h, w]])
                    
        save_ply(os.path.join(dst, "%s_pc.ply"%prefix), pointcloud_vis)

        
    print("max depth: ", max_height)
    return heights

filenames = os.listdir(depth_src)
# convert_depth_info(filenames)

thread_num = 10
file_num_per_thread = len(filenames) // thread_num

input_list = []
for i in range(thread_num):
    input_list.append(filenames[i*file_num_per_thread:(i+1)*file_num_per_thread])

height = []
# convert_depth_info(filenames)
with multiprocessing.Pool(len(input_list)) as pool:
    for threads_heigth in pool.imap_unordered(convert_depth_info, input_list):
        height += threads_heigth
