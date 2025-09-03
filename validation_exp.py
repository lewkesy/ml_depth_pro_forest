from PIL import Image
import depth_pro
import numpy as np
import os
from tqdm import tqdm
import torch
from IPython import embed
import cv2
import trimesh

src = "/home/zhou1178/meshgpt-pytorch/experiment_data/data"
depth_dst = '/home/zhou1178/meshgpt-pytorch/experiment_data/depth'
obj_src = "/data/zhou1178/MeshGPT_TreeStructor_validation/meshes"

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms(device=torch.device("cuda:0"))
model.eval()

for cluster in os.listdir(src):
    
    os.makedirs(os.path.join(depth_dst, cluster), exist_ok=True)
    print(cluster)
    
    for filename in tqdm(os.listdir(os.path.join(src, cluster))):
        image_path = os.path.join(src, cluster, filename)
        prefix = filename.split('.')[0]
        image, _, f_px = depth_pro.load_rgb(image_path, resize_h=256)
        image = transform(image)
        
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"].detach().cpu().numpy()  # Depth in [m].
        
        # align the depth
        depth = depth.max() - depth
        
        rgb = cv2.imread(os.path.join(src, cluster, filename))
        rgb = cv2.resize(rgb, (256, 256))
        
        mask = 1 - ((rgb[:,:,0]==255) * (rgb[:,:,1]==255) * (rgb[:,:,2]==255))
        depth = depth * mask
        
        mesh = trimesh.load(os.path.join(obj_src, prefix[:-2]+".obj"), force='mesh')
        vertices = mesh.vertices
        depth_max = vertices[:, 1].max()
        
        depth = depth / depth.max() * depth_max
        depth = depth * mask
        
        mask_image = (depth != 0).astype(np.int8) * 255
        
        np.savez(os.path.join(depth_dst, cluster, prefix + '.npz'), **{'depth':depth})
        cv2.imwrite(os.path.join(depth_dst, cluster, prefix + '.png'), mask_image)
        
