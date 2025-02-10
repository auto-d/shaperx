import os 
import pandas as pd
import re 
import open3d as o3d
import copy 
import numpy as np

from open3d.visualization import rendering

class_map = {
    'L1' : 0, 
    'L2' : 1, 
    'L3' : 2, 
    'L4' : 3, 
    'L5' : 4, 
    'T1' : 5, 
    'T2' : 6, 
    'T3' : 7, 
    'T4' : 8, 
    'T5' : 9, 
    'T6' : 10, 
    'T7' : 11, 
    'T8' : 12, 
    'T9' : 13, 
    'T10' : 14, 
    'T11' : 15, 
    'T12' : 16, 
    'C1' : 17, 
    'C2' : 18, 
    'C3' : 19, 
    'C4' : 20, 
    'C5' : 21, 
    'C6' : 22, 
    'C7' : 23, 
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
                 
def load_mesh_metadata(path:str) -> pd.DataFrame: 
    """
    Retrieve metadata for downloaded mesh files in the provided directory
    """
    metadata = pd.DataFrame(columns=['filename', 'label'])
    for i, file in enumerate(os.listdir(path)): 
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


import vtk 
class Renderer(): 
    """
    Wrapper around vtk funcs to simplify manipulating a 3d model and writing 
    to file. Improves on a few shortcomings of the Open3d implementations above. 
    """
    def __init__(self, size): 

        self.renderer = vtk.vtkRenderer()        
        self.renderer.SetBackground(0.5, 0.5, 0.5)  # Gray (right?)

        # Add lighting
        light = vtk.vtkLight()
        light.SetPosition(1, 1, 1) 
        #light.SetFocalPoint(0, 0, 0)  
        #light.SetColor(1, 1, 1) 
        light.SetIntensity(1.0)
        self.renderer.AddLight(light)

        # Set up an aperture to write our scene to a 2d image
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(size, size)  

        # Create one of these thingies 
        self.w2i = vtk.vtkWindowToImageFilter()
        self.w2i.SetInput(self.render_window)
        # TODO: internet says we need this... verify and uncomment
        self.w2i.ReadFrontBufferOff()  # Ensure correct buffer is used

        # I/O objects 
        self.reader = vtk.vtkSTLReader()
        self.writer = vtk.vtkPNGWriter()
        self.writer.SetInputConnection(self.w2i.GetOutputPort())

        # Helper to map polygonal geometry to a render pipeline
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.reader.GetOutputPort())

        self.actor = None

    def load(self, file): 
        """
        Load an STL model
        """
            
        self.reader.SetFileName(file)
        self.reader.Update()

        # Create an internal representation of our 3d model. Note we
        # eject the previous model when we load a new one here
        if self.actor: 
            self.renderer.RemoveActor(self.actor)
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

        self.actor.GetProperty().SetColor(0.9, 0.9, 0.9)  # Light gray
        self.actor.GetProperty().SetSpecular(0.5)
        self.actor.GetProperty().SetSpecularPower(20)

        # Add and reset camera to fit the entire object
        self.renderer.AddActor(self.actor)        
        self.renderer.ResetCamera()

    def rotate(self, x, y, z): 
        """
        Rotate the model
        """
        if self.actor: 
            self.actor.RotateX(x)
            self.actor.RotateY(y)
            self.actor.RotateZ(z)
            #self.actor.Modified()
            self.render_window.Render() 

    def write(self, path): 
        """
        Write a PNG of dim size x size to disk at the provided path 
        """
        self.render_window.Render()        
        self.w2i.Modified()
        self.w2i.Update()        
        self.writer.SetFileName(path)        
        self.writer.Write()

