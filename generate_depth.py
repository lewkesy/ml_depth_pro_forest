from PIL import Image
import depth_pro
import numpy as np
import os
from tqdm import tqdm
import torch
from IPython import embed

src = "/data/zhou1178/MeshGPT_TreeStructor_Large_Dataset"
depth_dst = '/data/zhou1178/MeshGPT_TreeStructor_Large_Dataset/depth'

os.makedirs(depth_dst, exist_ok=True)

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms(device=torch.device("cuda:0"))
model.eval()

cnt = 0
prefix_list = []
image_list = []

for filename in tqdm(os.listdir(os.path.join(src, 'rgb'))):
    # Load and preprocess an image.
    
    if cnt >= 2:
        # Run inference.
        image_list = torch.stack(image_list)
        prediction = model.infer(image_list, f_px=f_px)
        depth = prediction["depth"].detach().cpu().numpy()  # Depth in [m].
        # focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        
        for idx in range(len(prefix_list)):
            np.savez(os.path.join(depth_dst, prefix_list[idx]+'.npz'), **{'depth':depth[idx]})
        
        prefix_list = []
        image_list = []
        cnt = 0

    prefix = filename.split('.')[0][:-2]
    image_path = os.path.join(src, 'rgb', filename)
    image, _, f_px = depth_pro.load_rgb(image_path, resize_h=256)
    image = transform(image)
    
    prefix_list.append(prefix)
    image_list.append(image)
    # embed()

    cnt += 1


if cnt != 0:
    image_list = torch.stack(image_list)
    prediction = model.infer(image_list, f_px=f_px)
    depth = prediction["depth"].detach().cpu().numpy()  # Depth in [m].
    # focallength_px = prediction["focallength_px"]  # Focal length in pixels.
    
    for idx in range(len(prefix_list)):
        np.savez(os.path.join(depth_dst, prefix_list[idx]+'.npz'), **{'depth':depth[idx]})
    
    prefix_list = []
    image_list = []
    cnt = 0
        

    
    