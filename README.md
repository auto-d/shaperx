![Duke AIPI Logo](https://storage.googleapis.com/aipi_datasets/Duke-AIPI-Logo.png)

# Shaperx AIPI 540 Project #1

[MedshapeNet Reference](https://arxiv.org/abs/2308.16139)

[Hosted Gradio App Location](https://7c7b056f60a7fed238.gradio.live/)

[Project Presentation](https://prodduke-my.sharepoint.com/personal/jjm126_duke_edu/_layouts/15/stream.aspx?id=%2Fpersonal%2Fjjm126%5Fduke%5Fedu%2FDocuments%2FAIPI%2D540%2Fvision%5Fproject%5Fpresentation%2Emp4&ga=1&referrer=StreamWebApp%2EWeb&referrerScenario=AddressBarCopied%2Eview%2Ed64e213c%2De763%2D4f45%2D86aa%2D97d52f919d93)

## Introduction
This project utilizes the MedShapeNet database to develop and evaluate a machine learning classifier, a naïve approach, a deep learning and non-deep learning model for medical shape classification. All data used in this study are de-identified and publicly available, ensuring the privacy and confidentiality of individuals are protected. MedShapeNet database is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. 

## Problem Statement
Problem Statement: This project aims to develop a model using Medshapenet vertebrae data to classify medical images into a set of pre-defined medical categories, comparing the performance of a naive, non-deep learning model, and a deep learning model (e.g., Convolutional Neural Network). Accurate, unbiased classification is integral in determining diagnosis and pathology, and may be useful in terms of allocating resources and advocating for next-step treatment options.

## Data Source
[Source](https://arxiv.org/abs/2308.16139)

## Previous Literature

Li, J., et al., "Anatomy Completor: A Multi-class Completion Framework for 3D Anatomy Reconstruction," arXiv:2309.04956 (2023)

- Employs 3D auto-encoders and other deep learning models for mapping and reconstructing anatomical shapes using MedShapeNet database.

MedShapeNet [GitHub](https://github.com/GLARKI/MedShapeNet2.0)

- Project documentation and resources with stated intended uses for classification and reconstruction. ​
- Dedicated Python API, allowing for integration into ML/AI pipelines. 

**Theme**: Health, wellness and fitness

## Models Selected

### Naive Classifier

Naive approach :

- Histogram baseline of training corpus
- Chi-Square Distance measure for prediction 
   
### Traditional ML Pipeline (support vector machine)

SVM classifier:

- standardizes features into the model using StandardScaler
- stratifies category labels as best practice to minimize class imbalance while training
- Predicts the class using the trained svm model

### Deep Learning Pipeline (Convolutional Neural Network)

Neural-network-based approach : 

- PyTorch-based convolutional neural network (CNN)
- 2 to 5 convolutional layers depending on image size
- Convolution layers transition to fully connected

## Modeling Approach   

### Data Processing Pipeline

- Acquire meshes for all vertebrae
- Stratify and sample training, validation, and test sets
- Set meatadata for generation of input images
- Load into a scene, illuminate, orbit camera, and render
- Store image for all positions at configurable size
- Memorialize all metadata for the campaign for each data set

### Model Evaluation Process

 - Ensure class balance trhough stratified sampling
 - Hold out datasets for validation and test stages
 - Avoid cross validation due to size of dataset
 - Prefer top-K and confusion matrix given number of classes
 - Run each model through same dataset and apply metrics

### Metric Selection

1) Accuracy: Investigating performance on a per class basis
2) Confusion matrix: Visually assessing performance for each model
3) Top K: Assessing model improvement over 24 classes. 

### Development

For dependencies, see [requirements](requirements.txt). 

If working on the UI, run in gradio development mode: 

`gradio app/app.py`

⚡ The app expects to be run from the root of the repo, not the app directory. Mostly because we have the 'data' directory hardcoded.


### Deployment

The model can be demonstrated by deploying the included gradio application. 

1. cd <repo>/app
2. python app.py 
3. point browser at [https://3c7e4b51043a5486da.gradio.live/]

### Visual Interface 

The Gradio app allows the user to walk through the model pipeline, from downloading 3d models through classification and validation. 

![alt text](app.png)

## Usage 


## Evaluation

    # Figure 1 Confusion Matrix - Validation Set - Traditional ML Model

    # Figure 2 Classification Report - Validation Set - Non - Deep Learning Model

    # Figure 3 Classification Report - Test Set - Deep Learning Model

### Metric

While the precision and recall of individual classes is a useful metric for debugging our classification operation, we assess that the user really cares about overall *accuracy*. That is, the total number of correctly vs incorrectly classifed images. This metric is intuitive and concise in the context of vertebrae classification . 

## Results

Naive model: 
 - Accuracy: .07
 - Top-K: N/A
SVM:
 - Accuracy: .19
 - Top-K: .31
CNN:
 - Accuracy: .21
 - Top-K: .30

## Conclusion
 - Image Resolution: Task seems approachable, but low-resolution images don't betray enough detail to add rotation and scale invariance we need, higher-resolution deny training at larger scale NN requires to learn features ​

- Neural Network Training: Difficulty is high with so many variables​

- Variable Image Size: Using a configurable image size for the corpus and networks introduced unneeded complexity​

- Runtime: SVM and neural network resisted training efforts for the large (80K images @ 256 x 256)
## Outstanding Issues 

 - Hyper-parameter Search Space: Overall network architecture, kernel sizes, feature map depth, learning rate, momentum, elastic architecture for variable image dimensions create a large search space​
- Divergent Train/Val Scores: NN-based methods converged, but memorized training data and often resisted improvement on validation sets for tens of examples per class​
- Per-resolution hyper-parameters: At 32x32 we can converge with LR @ 0.002. But at 128x128 resolution, we will reliably fail unless LR is 0.001 or below.
## Ethics Statement

This project utilizes the MedShapeNet database to develop and evaluate a machine learning classifier, a naïve approach, a deep learning and non-deep learning model for medical shape classification. All data used in this study are de-identified and publicly available, ensuring the privacy and confidentiality of individuals are protected. MedShapeNet database is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. 
