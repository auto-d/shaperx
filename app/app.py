import gradio as gr
import subprocess
import os 
import pandas as pd
import re 
import dataset
import torch
import torchvision 
import torchvision.transforms as transforms
import cnn 

## Disclaimer: the use of globals here is mildly annoying, but passing all 
## this state around in gradio isn't super elegant, so we opt to suffer these

# Import/export location for 3d models
data_dir = 'data'

# Generate/retrieve experiment runs into this folder
experiments_dir = 'experiments'
experiment_no = 1

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

# Our CNN
net = None

def setup_app_dirs(): 
    """
    Find an experiment number that hasn't been used
    """
    global experiment_no 
    
    while True: 
        path = os.path.join(experiments_dir, str(experiment_no))
        if os.path.exists(path):
            experiment_no += 1
        else:                 
            os.makedirs(path) 
            break 
    
    if not os.path.exists(data_dir): 
        os.makedirs(data_dir)

def get_experiment_dir(): 
    """
    Accessor to wrangle users of the experiment dir
    """
    global experiment_no
    return os.path.join(experiments_dir,str(experiment_no))

def download_dataset(filename): 
    """
    Retrieve the datasets based on a manifest of URLs
    """    
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

def generate_split_images(samples, path, size, angle_increment): 
    """
    Generate an image set for one of our splits
    """
    
    if not os.path.exists(path): 
        os.makedirs(path) 

    images = pd.DataFrame(columns=['source', 'file', 'label'])
    insert_at = 0
    examples = []

    renderer = dataset.Renderer(size)

    # Iterate over the samples we selected from the dataset
    for row in samples.itertuples(): 
        id = row[0]
        file = row[1]
        label = row[2]
        
        renderer.load(os.path.join(data_dir,file))

        # Iterate over the camera angles implied by the angle step/stride
        
        X = range(0, 360, angle_increment)
        Y = range(0, 360, angle_increment)
        Z = range(0, 360, angle_increment)
        for x in X: 
            for y in Y: 
                for z in Z: 

                    # Write, taking care to distinguish the full path for examples
                    # and the path-free file name destined for pytorch
                    image_file = f"{id}-{label}-{size}-{x}-{y}-{z}.png"
                    image_path = os.path.join(path,image_file)

                    renderer.write(image_path)
                    print(f"Image written to {image_path}")
                    
                    images.loc[insert_at] = { 
                        'source': file, 
                        'file' : image_file, 
                        'label': dataset.get_class_index(label)
                        }
                    insert_at += 1
                        
                    if len(examples) < 20: 
                        examples.append(image_path)
                    
                    # Apply *relative* rotation 
                    renderer.rotate(0,0,angle_increment)                
                # <- Don't screw up the indentation here
                renderer.rotate(0,angle_increment,0)            
            # <- ...or here 
            renderer.rotate(angle_increment,0,0)

    return images, examples

def get_trn_csv(): 
    return os.path.join(get_experiment_dir(),'train.csv')

def get_tst_csv(): 
    return os.path.join(get_experiment_dir(),'test.csv')

def get_val_csv(): 
    return os.path.join(get_experiment_dir(),'val.csv')

def generate_images(size_pixels, angle_increment):
    """
    Use our 3d models to permute and emit images for each split
    """
    global images_trn 
    global images_tst
    global images_val 

    path = get_experiment_dir()

    # Generate imagesets for each split
    images_trn, examples = generate_split_images(mesh_trn, path, size_pixels, angle_increment)
    images_tst, _ = generate_split_images(mesh_tst, path, size_pixels, angle_increment)
    images_val, _ = generate_split_images(mesh_val, path, size_pixels, angle_increment)

    # Our dataframe has some extra information that needs to be ejected before we 
    # create the pytorch-esqe annotations file. This memorializes the splits and permits us to 
    # pick up later with these labelsets.
    annotations = images_trn.drop(labels='source', axis='columns')
    annotations.to_csv(get_trn_csv(), index=False)
    annotations = images_tst.drop(labels='source', axis='columns')
    annotations.to_csv(get_tst_csv(), index=False)
    annotations = images_val.drop(labels='source', axis='columns')
    annotations.to_csv(get_val_csv(), index=False)

    return examples

def train_model(): 
    """
    Train a vanilla CNN to classify using the training set
    """
    global net 
    
    # Create the CNN
    net = cnn.Net() 

    # Instantiate the pytorch loader with our custom DataSet
    loader = cnn.get_data_loader(get_trn_csv(), get_experiment_dir(), batch_size=2) 
    
    # Train 
    result = cnn.train(loader, net)

    return result

def classify(image): 
    global net
    
    # TODO: we need to pass the class label back here, not the logits
    prediction = net(image)
    
    return prediction

def test_model(): 
    pass

def main(): 
    setup_app_dirs() 

    demo = gr.Blocks()
    with demo: 

        # Header 
        gr.Markdown(value="# 🦴 ShapeRx Vision Pipeline")
        gr.Markdown(value=f"Experiment #**{experiment_no}**")
        # Load
        
        # This works, and can be used but is here to illustrate the process for the demo
        # or simplify download, not both. :)  I.e. we shouldn't be downloading the dataset
        # during the demo (takes forever) 
        gr.Markdown(value="## 📦 Load")
        gr.Markdown(value="### Update or retrieve new stereo lithography (STL) files.")
        with gr.Group(): 
            manifest_input = gr.Textbox(label="Dataset manifest:", value="MedShapeNetDataset_vertebrae_min_labeled.txt")
            with gr.Row(): 
                manifest_button = gr.Button("Download")
                manifest_output = gr.Markdown()
            manifest_button.click(fn=download_dataset, inputs=manifest_input, outputs=manifest_output)

        # View Sample
        gr.Markdown(value="### Review downloaded models")
        with gr.Group():             
            with gr.Row(): 
                height = 250
                sample_mesh_input = gr.FileExplorer(file_count='single', root_dir=data_dir, max_height=height)
                sample_mesh_output = gr.Model3D(height=height, label='Selected sample')
            sample_mesh_button = gr.Button("Render sample")
            sample_mesh_button.click(fn=retrieve_sample_mesh, inputs=sample_mesh_input, outputs=sample_mesh_output)

        # Process
        gr.Markdown(value="## ♻️ Process")
        gr.Markdown(value="### Process downloaded 3d models")
        with gr.Group(): 
            metadata_button = gr.Button("Extract metadata")
            metadata_output = gr.Dataframe(max_height=200)
            metadata_button.click(fn=load_metadata, inputs=None, outputs=metadata_output)

        # Mesh Sampling    
        gr.Markdown(value="### Select sources for model pipeline")
        gr.Markdown(value=f"We'll sample for class (we have {classes} classes)")
        with gr.Group():             
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
        gr.Markdown(value="## 🪄 Generate")
        gr.Markdown(value="Permute the various facets of the rendering and generate an image set")    
        with gr.Group():                         
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
        gr.Markdown(value="## ⚙️ Train")
        with gr.Group():            
            train_button = gr.Button("Train CNN")
            with gr.Row(): 
                gr.Markdown(value="Training result:")
                train_result = gr.Markdown()

            #TODO: adjust hyperparameters from the UI?
            train_button.click(fn=train_model, inputs=None, outputs=train_result)

        # Model Testing
        # TODO: run test set through model and compute various values... 
        # plot metrics at each epoch with gd.LinePlot

    demo.launch(share=True)

if __name__ == "__main__":
    main()