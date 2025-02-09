import gradio as gr
import subprocess
import os 
import pandas as pd
import re 
import dataset

## Disclaimer: the use of globals here is mildly annoying, but passing all 
## this state around in gradio isn't super elegant, so we opt to suffer these

# Import/export location for 3d models
data_dir = 'data'

# Generate/retrieve experiment runs into this folder
experiments_dir = 'experiments'

# Metadata about all the 3d models we know about
metadata = None

# Metadata about the *sampled* models for each split
mesh_trn = None
mesh_tst = None
mesh_val = None

# Metadata about the images we generate for each split
images_trn = None
images_tst = None
images_val = None 

# Count of classes in the dataset
classes = len(dataset.class_map)

def download_dataset(filename): 
    """
    Retrieve the datasets based on a manifest of URLs
    """
    if not os.path.exists(data_dir): 
        os.makedirs(data_dir) 

    subprocess.run(
        ["wget", "--content-disposition"," --trust-server-names", "--no-clobber","-i",filename], 
        cwd=data_dir,
        )
    
    return "Done!"

def retrieve_sample_mesh(file): 
    """
    Retrieve a sample mesh
    """
    # No processing required, our inputs = outputs for the UI widgets
    return file

def load_metadata(): 
    """
    Scan the data directory for all available 3d mesh sources (of vertebrae)
    """
    global metadata
    metadata = dataset.load_mesh_metadata(data_dir)
    groups = metadata.groupby(by='label')
    df = groups.size().reset_index()
    df.columns = ['labels', 'count']
    return df

def sample_metadata(trn_n, tst_n, val_n): 
    """
    For each split, sample the provided number of rows for each class and 
    return as three separate dataframes for the user to validate the operation
    """
    global metadata
    global mesh_trn
    global mesh_tst
    global mesh_val
    
    if isinstance(metadata, pd.DataFrame):
        mesh_trn, mesh_tst, mesh_val = dataset.split_meshes(
            metadata, 
            trn_n, 
            tst_n, 
            val_n,
        )

        groups = mesh_trn.groupby(by='label')
        counts_trn = groups.count().reset_index()
        counts_trn.columns = ['label', 'count']
        
        groups = mesh_tst.groupby(by='label')
        counts_tst = groups.count().reset_index()
        counts_tst.columns = ['label', 'count']

        groups = mesh_val.groupby(by='label')
        counts_val = groups.count().reset_index()
        counts_val.columns = ['label', 'count']
        
        return counts_trn, counts_tst, counts_val

def compute_image_count(angle_increment):  
    """
    Estimate the number of images that will be generated for the given angle
    """
    if isinstance(mesh_trn, pd.DataFrame):
        viewpoints_per_axis = 360/angle_increment    
        count = \
            mesh_trn.shape[0] * viewpoints_per_axis**3 \
            + mesh_tst.shape[0] * viewpoints_per_axis**3 \
            + mesh_val.shape[0]* viewpoints_per_axis**3 
        return f"{int(count)}"

def generate_split_images(samples, dir, size_pixels, angle_increment): 
    """
    Generate an image set for one of our splits
    """

    # new implementation of load, rotate and save goes in here, in lieu of
    # below. BUT, make sure the new implementation returns a dataframe with 
    # the generated file and associated label for follow on model training
    
    if not os.path.exists(dir): 
        os.makedirs(dir) 

    images = pd.DataFrame(columns=['source', 'file', 'label'])
    insert_at = 0
    examples = []

    # Iterate over the samples we selected from the dataset
    for row in samples.itertuples(): 
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
                        path = os.path.join(experiments_dir,f"{id}-{label}-{size_pixels}-{x}-{y}-{z}.png")

                        dataset.save_image_mask(mesh=rotated, 
                                           h=size_pixels, 
                                           w=size_pixels, 
                                           png=path)
                        images.loc[insert_at] = { 'source': file, 'file' : path, 'label': label}
                        insert_at += 1
                         
                        if len(examples) < 20: 
                            examples.append(path)
        else: 
            print(f"Error: unable to load vertices for model {file}. Moving to next file.")

    return images, examples

def generate_images(size_pixels, angle_increment):
    """
    Use our 3d models to permute and emit images for each split
    """
    global images_trn 
    global images_tst
    global images_val 

    if not os.path.exists(experiments_dir): 
        os.makedirs(experiments_dir) 

    images_trn, examples = generate_split_images(mesh_trn, experiments_dir, size_pixels, angle_increment)
    images_tst, _ = generate_split_images(mesh_tst, experiments_dir, size_pixels, angle_increment)
    images_val, _ = generate_split_images(mesh_val, experiments_dir, size_pixels, angle_increment)

    # Provide examples of the training set for the user to visually inspect
    return examples

def train_model(): 
    pass

def test_model(): 
    pass

def batch_train(): 
    pass

demo = gr.Blocks()
with demo: 

    # Load
    
    # This works, and can be used but is here to illustrate the process for the demo
    # or simplify download, not both. :)  I.e. we shouldn't be downloading the dataset
    # during the demo (takes forever) 
    gr.Markdown(value="# 📦 Load")
    gr.Markdown(value="## Update or retrieve new stereo lithography (STL) files.")

    manifest_input = gr.Textbox(label="Dataset manifest:", value="MedShapeNetDataset_vertebrae_min_labeled.txt")
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
    gr.Markdown(value="## Select sources for model pipeline")
    gr.Markdown(value=f"We'll sample for class (we have {classes} classes)")
    with gr.Row():
        mesh_trn_slider = gr.Slider(label="Training sources", value=5, maximum=20, step=1)
        mesh_tst_slider = gr.Slider(label="Test sources", value=1, maximum=20, step=1)
        mesh_val_slider = gr.Slider(label="Validation sources", value=1, maximum=20, step=1)
    mesh_sample_button = gr.Button("Sample")
    with gr.Row(): 
        mesh_trn_output = gr.Dataframe(label="Training sources selected")
        mesh_tst_output = gr.Dataframe(label="Training sources selected")
        mesh_val_output = gr.Dataframe(label="Training sources selected")
    mesh_sample_button.click(
        fn=sample_metadata, 
        inputs=[
            mesh_trn_slider, 
            mesh_tst_slider, 
            mesh_val_slider,
        ],
        outputs=[
            mesh_trn_output,
            mesh_tst_output,
            mesh_val_output,
        ])

    # Image Generation
     
    # As above with loading, this is great for experimentation, but shouldn't be run 
    # during the demo with anything but trivial values since it will take hours to 
    # generate and can safely be skipped. 
    gr.Markdown()
    gr.Markdown(value="# 🪄 Generate")
    gr.Markdown(value="Permute the various facets of the rendering and generate an image set")    
    
    with gr.Row(): 
        image_size = gr.Slider(label="Image size (pixels)", value=64, minimum=32, maximum=256, step=32)
        image_angle = gr.Slider(label="Angle increments (degrees)", value=45, minimum=0, maximum=360, step=15)
    with gr.Row(): 
        image_count_text = gr.Markdown(value="Images to generate:")
        image_count = gr.Markdown(value="*Extract metadata and sample to see estimate*")
    image_angle.release(fn=compute_image_count, inputs=[image_angle], outputs=image_count)
    image_generate_button = gr.Button("Generate images")
    image_gallery_label = gr.Markdown(value="Training set examples:")
    image_gallery = gr.Gallery()

    image_generate_button.click(fn=generate_images, inputs=[image_size, image_angle], outputs=image_gallery)
    
    # Model Training 
    # TODO: 

    # Model Testing
    # TODO: 

demo.launch(share=False)