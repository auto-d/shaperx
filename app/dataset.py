import os 
import pandas as pd
import re 
import open3d as o3d
import copy 
import numpy as np
from open3d.visualization import rendering

class_map = {
    'L1' : 1, 
    'L2' : 2, 
    'L3' : 3, 
    'L4' : 4, 
    'L5' : 5, 
    'T1' : 6, 
    'T2' : 7, 
    'T3' : 8, 
    'T4' : 9, 
    'T5' : 10, 
    'T6' : 11, 
    'T7' : 12, 
    'T8' : 13, 
    'T9' : 14, 
    'T10' : 15, 
    'T11' : 16, 
    'T12' : 17, 
    'C1' : 18, 
    'C2' : 19, 
    'C3' : 20, 
    'C4' : 21, 
    'C5' : 22, 
    'C6' : 23, 
    'C7' : 24, 
}

def get_class_index(name:str): 
    """
    Map class name to index
    """ 
    name = name.upper()
    if name in class_map: 
        return class_map[name]
    
def get_class_label(index:int):
    """
    Map index to class name
    """
    for key, value in class_map.items():
        if index == value: 
            return key 

def decode_label(filename:str) -> str:
    """
    Retrieve a label from the vertebrae dataset mesh file name
    """
    match = re.search(".*vertebrae_?([A-Z][0-9]+).*.stl", filename)
    if match: 
        return match.group(1)
                 
def load_mesh_metadata(dir:str) -> pd.DataFrame: 
    """
    Retrieve metadata for downloaded mesh files in the provided directory
    """
    metadata = pd.DataFrame(columns=['filename', 'label'])
    for i, file in enumerate(os.listdir(dir)): 
        if file.endswith('stl'): 
            metadata.loc[i] = {'filename': file, 'label': decode_label(file)}

    return metadata

def sample_meshes(df, n=10):
    """
    Sample from the downloaded mesh files at a rate of n per 
    class (stratify)
    """
    samples = pd.DataFrame()
    for _class in df['label'].unique(): 
        class_samples = df[df['label'] == _class].sample(n=n)
        
        samples = pd.concat([samples, class_samples])

    return samples

def split_meshes(df, trn_n, tst_n, val_n):
    """
    Generate stratified subsets of the dataframe for provided
    test, train and validation counts. 
    """
    trn = pd.DataFrame()
    tst = pd.DataFrame()
    val = pd.DataFrame()

    # If we have enough data to generate the requested splits, 
    if isinstance(df, pd.DataFrame):
        samples = sample_meshes(df, n = trn_n + tst_n + val_n)
        for _class in df['label'].unique(): 
            class_samples = df[df['label'] == _class]
            not_val = class_samples.head(trn_n + tst_n)
            val = pd.concat([val, class_samples.tail(val_n)])
            trn = pd.concat([trn, not_val.head(trn_n)])
            tst = pd.concat([tst, not_val.tail(tst_n)])

    return trn, tst, val

def load_mesh(file): 
    """
    Load a mesh from disk and return an Open3d mesh object
    """
    if file.endswith('stl'): 
        return o3d.io.read_triangle_mesh(file)
        
def draw_mesh(mesh): 
    """
    Show an open3d mesh object
    """
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], width=192, height=108)

def rotate_mesh(mesh,x, y, z): 
    """
    Rotate an open3d mesh object
    """
    # https://www.open3d.org/docs/latest/tutorial/geometry/transformation.html#Rotation
    mesh_r = copy.deepcopy(mesh)
    
    # degrees -> radians 
    X = x / 180 * np.pi
    Y = y / 180 * np.pi 
    Z = z / 180 * np.pi
    R = mesh.get_rotation_matrix_from_xyz((X, Y, Z))
    mesh_r.rotate(R, center=(0, 0, 0))

    return mesh_r

def save_image_mask(mesh, h, w, png): 
    """
    Generate a PNG for the provided mesh of h x w pixels, silhouette only
    """
    # https://github.com/isl-org/Open3D/issues/1095
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=w, height=h)
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(png,do_render=True)    
    vis.destroy_window()