import numpy as np
from plyfile import PlyData, PlyElement

def save_mesh(fn, xyz, faces):
    with open(fn, 'w') as f:
        point_num = xyz.shape[0]
        face_num = faces.shape[0]
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % (point_num))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('element face %d\n' % (face_num))
        f.write('property list uchar int vertex_index\n')
        f.write('end_header\n')
        for i in range(point_num):
            f.write('%.6f %.6f %.6f\n' % (xyz[i][0], xyz[i][1], xyz[i][2]))
        for i in range(face_num):
            f.write('%d %d %d %d\n' % (faces[i][0], faces[i][1], faces[i][2] ,faces[i][3]))