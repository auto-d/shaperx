import gradio as gr
import subprocess
import os 
import pandas as pd
import re 
import dataset

data_dir = 'data'
metadata = None

def download_dataset(filename): 
    if not os.path.exists(data_dir): 
        os.makesirs(data_dir) 

    subprocess.run(
        ["wget", "--content-disposition"," --trust-server-names","-i",filename], 
        cwd=data_dir,
        )
    
    return "Done!"

def load_metadata(): 
    global metadata
    metadata = dataset.load_mesh_metadata(data_dir)
    groups = metadata.groupby(by='label')
    df = groups.size().reset_index()
    df.columns = ['labels', 'count']
    return df

def sample_metadata(n): 
    global metadata
    samples = dataset.sample_meshes(metadata, n)
    groups = samples.groupby(by='label')
    counts = groups.count().reset_index()
    counts.columns = ['label', 'count']
    return counts

demo = gr.Blocks()
with demo: 

    file_input = gr.FileExplorer()
        
    # Load
    manifest_input = gr.Textbox(label="Dataset manifest:", value="MedShapeNetDataset_vertebrae_labeled.txt")
    with gr.Row(): 
        manifest_button = gr.Button("Download")
        manifest_output = gr.Markdown()
    manifest_button.click(fn=download_dataset, inputs=manifest_input, outputs=manifest_output)

    # Extract
    metadata_button = gr.Button("Extract metadata")
    metadata_output = gr.Dataframe(max_height=200)
    metadata_button.click(fn=load_metadata, inputs=None, outputs=metadata_output)

    # Mesh Sampling    
    with gr.Row():
        mesh_sample_button = gr.Button("Sample")
        mesh_sample_slider = gr.Slider(label="Sample count")
    mesh_sample_output = gr.Dataframe()
    mesh_sample_button.click(fn=sample_metadata, inputs=mesh_sample_slider, outputs=mesh_sample_output)

demo.launch(share=False)