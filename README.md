# ShapeRx

Classification of anatomical shapes. 

**Theme**: Health, wellness and fitness

## Design

1. Naive approach : B&w histograms? 
2. Classic machine learning approach : Support Vector Machine (SVM) classifier
3. Neural-network-based approach : PyTorch-based convolutional neural network (CNN)

## Development

For dependencies, see [requirements](requirements.txt). 

If working on the UI, run in gradio development mode: 

`gradio app/app.py`

⚡ The app expects to be run from the root of the repo, not the app directory. Mostly because we have the 'data' directory hardcoded.

## Testing 

TODO

## Deployment

The model can be demonstrated by deploying the included gradio application. 

1. cd <repo>/app
2. python app.py 
3. point browser at http://127.0.0.1:7860

### Interface 

The Gradio app allows the user to walk through the model pipeline, from downloading 3d models through classification and validation. 

![alt text](app.png)

## Outstanding Issues 

- ❗ a number of files are not loading into the mesh, warning printed is shown below. for now we are ignoring, but we should process the whole dataset to remove these and then resample, otherwise we'll have unintended class imbalance
  > [Open3D WARNING] Unable to load file data/001200_vertebrae.stl with ASSIMP: Unable to open file "data/001200_vertebrae.stl".
- ❗ we aren't inducing any scale or lighting changes in our images, remedy that! (?)