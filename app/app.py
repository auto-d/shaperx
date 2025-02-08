import gradio as gr
import subprocess
import os 
import pandas as pd
import re 
import dataset

data_dir = 'data'
experiment_dir = 'experiments'
metadata = None
mesh_samples = None
images = None
# TODO: comments... 

def download_dataset(filename): 
    if not os.path.exists(data_dir): 
        os.makesirs(data_dir) 

    subprocess.run(
        ["wget", "--content-disposition"," --trust-server-names", "--no-clobber","-i",filename], 
        cwd=data_dir,
        )
    
    return "Done!"

def retrieve_sample_mesh(file): 
    return file

def load_metadata(): 
    global metadata
    metadata = dataset.load_mesh_metadata(data_dir)
    groups = metadata.groupby(by='label')
    df = groups.size().reset_index()
    df.columns = ['labels', 'count']
    return df

def sample_metadata(n): 
    global metadata
    global mesh_samples
    
    if isinstance(metadata, pd.DataFrame):
        mesh_samples = dataset.sample_meshes(metadata, n)
        groups = mesh_samples.groupby(by='label')
        counts = groups.count().reset_index()
        counts.columns = ['label', 'count']
        return counts

def show_mesh(mesh):
    # https://www.gradio.app/guides/how-to-use-3D-model-component
    pass

def compute_image_count(angle_increment):  
    if isinstance(mesh_samples, pd.DataFrame):
        viewpoints_per_axis = 360/angle_increment    
        return f"{int(mesh_samples.shape[0] * viewpoints_per_axis**3)}"

def generate_images(size_pixels, angle_increment):
    
    images = pd.DataFrame(columns=['source', 'file', 'label'])
    insert_at = 0
    examples = []

    for row in mesh_samples.itertuples(): 
        id = row[0]
        file = row[1]
        label = row[2]
        
        sample = dataset.load_mesh(os.path.join(data_dir,file))
        if len(sample.vertices) > 0: 
            X = range(0, 360, angle_increment)
            Y = range(0, 360, angle_increment)
            Z = range(0, 360, angle_increment)
            for x in X: 
                for y in Y: 
                    for z in Z: 
                        rotated = dataset.rotate_mesh(sample, x, y, z)                        
                        path = os.path.join(experiment_dir,f"{id}-{label}-{size_pixels}-{x}-{y}-{z}.png")

                        dataset.save_image(mesh=rotated, 
                                           h=size_pixels, 
                                           w=size_pixels, 
                                           png=path)
                        images.loc[insert_at] = { 'source': file, 'file' : path, 'label': label}
                        insert_at += 1
                         
                        if len(examples) < 20: 
                            examples.append(path)
        else: 
            print(f"Error: unable to load vertices for model {file}. Moving to next file.")

    return examples

def split_data(): 
    pass

def train_model(): 
    pass

def test_model(): 
    pass

def batch_train(): 
    pass

demo = gr.Blocks()
with demo: 

    # Load
    gr.Markdown(value="# 📦 Load")
    gr.Markdown(value="## Update or retrieve new stereo lithography (STL) files.")

    manifest_input = gr.Textbox(label="Dataset manifest:", value="MedShapeNetDataset_vertebrae_labeled.txt")
    with gr.Row(): 
        manifest_button = gr.Button("Download")
        manifest_output = gr.Markdown()
    manifest_button.click(fn=download_dataset, inputs=manifest_input, outputs=manifest_output)

    # View Sample
    gr.Markdown(value="## Review downloaded models")
    with gr.Row(): 
        height = 250
        sample_mesh_input = gr.FileExplorer(file_count='single', root_dir=data_dir, max_height=height)
        sample_mesh_output = gr.Model3D(height=height, label='Selected sample')
    sample_mesh_button = gr.Button("Render sample")
    sample_mesh_button.click(fn=retrieve_sample_mesh, inputs=sample_mesh_input, outputs=sample_mesh_output)

    # Process
    gr.Markdown(value="# ♻️ Process")
    gr.Markdown(value="## Process downloaded 3d models")
    metadata_button = gr.Button("Extract metadata")
    metadata_output = gr.Dataframe(max_height=200)
    metadata_button.click(fn=load_metadata, inputs=None, outputs=metadata_output)

    # Mesh Sampling    
    gr.Markdown(value="## Conduct stratified sampling")
    with gr.Row():
        mesh_sample_slider = gr.Slider(label="Sample count", value=5, maximum=25)
        mesh_sample_button = gr.Button("Sample")        
    mesh_sample_output = gr.Dataframe()
    mesh_sample_button.click(fn=sample_metadata, inputs=mesh_sample_slider, outputs=mesh_sample_output)

    # Image Generation
    gr.Markdown()
    gr.Markdown(value="# 🪄 Generate")
    gr.Markdown(value="Permute the various facets of the rendering and generate an image set")    
    
    with gr.Row(): 
        image_size = gr.Slider(label="Image size (pixels)", value=64, minimum=32, maximum=256, step=32)
        image_angle = gr.Slider(label="Angle increments (degrees)", value=45, minimum=2, maximum=180, step=15)
    with gr.Row(): 
        image_count_text = gr.Markdown(value="Images to generate:")
        image_count = gr.Markdown(value="*Extract metadata and sample to see estimate*")
    image_angle.release(fn=compute_image_count, inputs=[image_angle], outputs=image_count)
    image_generate_button = gr.Button("Generate images")
    image_gallery = gr.Gallery()

    image_generate_button.click(fn=generate_images, inputs=[image_size, image_angle], outputs=image_gallery)
    

demo.launch(share=False)