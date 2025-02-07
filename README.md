# ShapeRx

Classification of anatomical shapes. 

Theme: Healt, wellness and fitness

## Design

1. Naive approach : B&w histograms? 
2. Non-deep learning approach : ?
3. NN-based approach : PyTorch-based convolutional neural network 

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

## Outstanding Issues 

- ❗ renderer isn't using a light source, so image look dull and featureless - implement a new render function to allow changing the light source and incorporate into a future training run
- ❗ a number of files are not loading into the mesh, warning printed is shown below. for now we are ignoring, but we should process the whole dataset to remove these and then resample, otherwise we'll have unintended class imbalance
  > [Open3D WARNING] Unable to load file data/001200_vertebrae.stl with ASSIMP: Unable to open file "data/001200_vertebrae.stl".
- ❗ we aren't inducing any scale change in our images, remedy that!