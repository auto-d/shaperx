import gradio as gr
import subprocess
import os 
import pandas as pd
import numpy as np
import re 
import mesh_dataset
import image_dataset
import torch
import torchvision 
import torchvision.transforms as transforms
import cnn 
import svm
import naive 
from sklearn.metrics import classification_report, accuracy_score

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
classes = len(mesh_dataset.class_map)

# Our models
naive_model = None 
svm_model = None
cnn_model = None

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

def list_experiments(): 
    """
    Inventory experiments
    """
    dirs = os.listdir('experiments')
    experiments = [int(dir) for dir in dirs if dir.isdigit()]
    experiments.sort()
    return experiments

def change_experiment(number:int): 
    """
    Update the current experiment, returns a textual representation to simplify display
    """
    global experiment_no 
    
    if type(number) != int: 
        raise ValueError(f"Experiment must be an integer value! Got {type(number)}")
    
    if number not in list_experiments(): 
        raise ValueError(f"Experiment {number} not a known experiment number!")
        
    experiment_no = number  

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
    metadata = mesh_dataset.load_mesh_metadata(data_dir)
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
        mesh_trn, mesh_tst, mesh_val = mesh_dataset.split_meshes(
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

def get_trn_csv(): 
    return os.path.join(get_experiment_dir(),'train.csv')

def get_tst_csv(): 
    return os.path.join(get_experiment_dir(),'test.csv')

def get_val_csv(): 
    return os.path.join(get_experiment_dir(),'val.csv')

def generate_image_set(size_pixels, angle_increment):
    """
    Use our 3d models to permute and emit images for each split
    """
    global images_trn 
    global images_tst
    global images_val 

    path = get_experiment_dir()

    # Generate imagesets for each split
    images_trn, examples = mesh_dataset.generate_image_set(mesh_trn, data_dir, path, size_pixels, angle_increment)
    images_tst, _ = mesh_dataset.generate_image_set(mesh_tst, data_dir, path, size_pixels, angle_increment)
    images_val, _ = mesh_dataset.generate_image_set(mesh_val, data_dir, path, size_pixels, angle_increment)

    mesh_dataset.save_image_set(images_trn, get_trn_csv())
    mesh_dataset.save_image_set(images_tst, get_tst_csv())
    mesh_dataset.save_image_set(images_val, get_val_csv())

    return examples

def load_image_sets(): 
    """
    Retrieve saved image metadata off disk for our three splits if needed
    """
    global images_trn 
    global images_tst
    global images_val 

    images_trn = mesh_dataset.load_image_set(get_trn_csv())
    images_tst = mesh_dataset.load_image_set(get_tst_csv())
    images_val = mesh_dataset.load_image_set(get_val_csv())

def train_naive_model(): 
    """
    Prepare a histogram classifier 
    """
    global images_trn
    global naive_model

    load_image_sets() 

    X, y = naive.load_dataset(images_trn, get_experiment_dir())
    
    naive_model = naive.NaiveEstimator()
    naive_model.fit(X, y) 

    path = naive.save_model(naive_model, get_experiment_dir())

    return f"Fit model on {len(X)} images. Model written to {path}."

def train_svm_model(): 
    """
    Train a vanilla CNN to classify using the training set
    """
    global images_trn
    global svm_model

    load_image_sets() 

    X, y = svm.load_dataset(images_trn, get_experiment_dir())

    svm_model = svm.SvmEstimator()
    svm_model.fit(X, y)

    path = svm.save_model(svm_model, get_experiment_dir())

    return f"Fit model on {len(X)} images. Model written to {path}."
    
def train_cnn_model(): 
    """
    Train a vanilla CNN to classify using the training set
    """
    global images_trn
    global cnn_model 
    
    load_image_sets() 

    # Retrieve the image height/width
    loader = cnn.get_data_loader(get_trn_csv(), get_experiment_dir(), batch_size=1) 
    shape = loader.dataset[0][0].shape 
    width = shape[2]

    cnn_model = cnn.Net(width) 

    loss_history = cnn.train(loader=loader, model=cnn_model, loss_interval=20, epochs=10, lr=0.002, momentum=0.1)

    path = cnn.save_model(cnn_model, get_experiment_dir())

    return f"Fit CNN on {len(loader.dataset)} images. Model written to {path}."

def classify_naive(model, imageset): 
    """
    Classify an imageset with our naive model
    """   
    X, _ = naive.load_dataset(imageset, get_experiment_dir())

    preds = model.predict(X)
    
    return preds

def classify_svm(model, imageset): 
    """
    Classify an image with our classical ML model 
    """   
    X, _ = svm.load_dataset(imageset, get_experiment_dir())

    preds = model.predict(X) 

    return preds

def classify_cnn(model, imageset): 
    """
    Classify an image with our neural network 
    """
    loader = cnn.get_data_loader(imageset, get_experiment_dir(), batch_size=1, shuffle=False)
    preds = cnn.predict(loader, model)
    
    return preds

def score(y_true, y):
    """
    Generically score a set of predictions against provided ground-truth
    """
    labels = list(mesh_dataset.class_map.values())
    label_names = list(mesh_dataset.class_map.keys())

    result = "" 
    accuracy = accuracy_score(y_true, y)
    
    result += f"Validation accuracy: {accuracy:.4f}"
    result += "\nClassification Report:"
    result += classification_report(y_true, y, labels=labels, target_names=label_names)
    
    return result

def evaluate(): 
    """
    Run the validation data through the models and report a winner
    """
    global naive_model 
    global svm_model 
    global cnn_model 
    global images_val

    load_image_sets() 

    y = images_val.label.to_numpy()

    naive_model = naive.load_model(get_experiment_dir())
    naive_preds = classify_naive(naive_model, images_val)
    result = score(y, naive_preds)
    
    svm_model = svm.load_model(get_experiment_dir())
    svm_preds = classify_svm(svm_model, images_val)
    result += score(y, svm_preds)

    cnn_model = cnn.load_model(get_experiment_dir()) 
    cnn_preds = classify_cnn(cnn_model, images_val)
    result += score(y, cnn_preds)

    return result

def main(): 
    global experiment_no
    setup_app_dirs() 

    #TODO: hard-coded for testing, remove
    # 82 - small dataset for testing @32 pixels
    # 25 - small dataset for testing  @256
    # 21 - large dataset for training @256
    change_experiment(82)

    demo = gr.Blocks()
    with demo: 

        # Header         
        gr.Markdown(value="# 🦴 ShapeRx Vision Pipeline")
        experiment_picker = gr.Dropdown(choices=list_experiments(), value=experiment_no, label='Experiment', interactive=True)
        experiment_picker.change(fn=change_experiment, inputs=[experiment_picker])

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
        gr.Markdown(value=f"We'll sample for each class (we have {classes} classes)")
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

            image_generate_button.click(fn=generate_image_set, inputs=[image_size, image_angle], outputs=image_gallery)
        
        # Model Training 
        gr.Markdown(value="## ⚙️ Train")
        with gr.Group():            
            with gr.Row(): 
                train_naive_button = gr.Button("Train Naive")
                train_naive_result = gr.Markdown()
            with gr.Row(): 
                train_svm_button = gr.Button("Train SVM")                
                train_svm_result = gr.Markdown()
            with gr.Row(): 
                train_cnn_button = gr.Button("Train CNN")                
                train_cnn_result = gr.Markdown()

            train_naive_button.click(fn=train_naive_model, inputs=None, outputs=train_naive_result)
            train_svm_button.click(fn=train_svm_model, inputs=None, outputs=train_svm_result)
            train_cnn_button.click(fn=train_cnn_model, inputs=None, outputs=train_cnn_result)

        # Model Validation 
        gr.Markdown(value="## 🧪 Test")
        with gr.Group():            
            evaluate_button = gr.Button("Evaluate")
            evaluate_result = gr.Markdown(value="Waiting for evaluation...")

        evaluate_button.click(fn=evaluate, inputs=None, outputs=evaluate_result)

    demo.launch(share=False)

if __name__ == "__main__":
    main()