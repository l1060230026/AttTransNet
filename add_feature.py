import open3d as o3d
import os
import numpy as np
from tqdm import tqdm

# root_path = 'data/stanford_indoor3d'
root_path = 'data/craslab3d'
files = os.listdir(root_path)

for file in tqdm(files):
    file_path = os.path.join(root_path, file)
    data = np.load(file_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:,:3])
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    data = np.hstack([data[:,:6], normals, data[:,-1:]])
    np.save(file_path, data)