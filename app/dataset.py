import os 
import pandas as pd
import re 
import open3d as o3d
import copy 
import numpy as np
from open3d.visualization import rendering

# TODO: comments... 

def decode_label(filename:str) -> str:
    match = re.search(".*vertebrae_?([A-Z][0-9]).*.stl", filename)
    if match: 
        return match.group(1)
                 
def load_mesh_metadata(dir:str) -> pd.DataFrame: 
    metadata = pd.DataFrame(columns=['filename', 'label'])
    for i, file in enumerate(os.listdir(dir)): 
        if file.endswith('stl'): 
            metadata.loc[i] = {'filename': file, 'label': decode_label(file)}

    return metadata

def sample_meshes(df, n=10):
    
    samples = pd.DataFrame()
    for _class in df['label'].unique(): 
        class_samples = df[df['label'] == _class].sample(n=n)
        
        samples = pd.concat([samples, class_samples])

    return samples

def load_mesh(file): 
    if file.endswith('stl'): 
        return o3d.io.read_triangle_mesh(file)
        
def draw_mesh(mesh): 
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], width=192, height=108)

def rotate_mesh(mesh,x, y, z): 
    # https://www.open3d.org/docs/latest/tutorial/geometry/transformation.html#Rotation
    mesh_r = copy.deepcopy(mesh)
    
    # degrees -> radians 
    X = x / 180 * np.pi
    Y = y / 180 * np.pi 
    Z = z / 180 * np.pi
    R = mesh.get_rotation_matrix_from_xyz((X, Y, Z))
    mesh_r.rotate(R, center=(0, 0, 0))

    return mesh_r

def save_image(mesh, h, w, png): 
    # https://github.com/isl-org/Open3D/issues/1095
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=w, height=h)
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(png,do_render=True)    
    vis.destroy_window()