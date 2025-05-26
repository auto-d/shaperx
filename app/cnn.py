#
# Adapated from PyTorch image classifier tutorial - https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#

from dataset import class_map

import os
import pandas as pd
import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

class Net(nn.Module):
    """
    Basic convolutional neural network derived from the pytorch tutorial 
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    def __init__(self):
        super().__init__()
                
        # Input channels, output channels, kernel height/width
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)         
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(class_map))

    def forward(self, x):
        # 1. Convolution + pooling
        #
        # Each filter here (conv_2d out_channels) is a unique opportunity to 
        # learn a useful pattern in the input. The filter count used is based on 
        # the pytorch example, but if we had the compute we would want to do a 
        # search over these values (as with most other network parameters here)!
        #
        # input:       3 x 32^2 
        # conv @ 6, 5: 6 x 28^2
        # pool @ 2x2:  6 x 14^2
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # 2. Convolution + pooling
        # input:       6 x 14^2 
        # conv @ 16,5: 16 x 10^2
        # pool @ 2x2:  16 x 5^2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # 3. Reshape
        # input:       16 x 5^2
        # flatten @ 1: 1 x 400
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        # 4. Fully-connected
        # input:       1 x 400
        # linear:      1 x 120
        x = self.fc1(x)
        x = F.relu(x)
        
        # 5. Fully-connected 
        # input:       1 x 120
        # linear:      1 x 84
        x = self.fc2(x)
        x = F.relu(x)
            
        # 6. Fully-connected output 
        # input:       1 x 84
        # linear:      1 x 24 (vertebrae/classes)
        x = self.fc3(x)
        
        return x

class ShapeRxDataset(torch.utils.data.Dataset): 
    """
    Custom pytorch-compatible dataset. Adapted from 
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None): 

        self.img_labels = pd.read_csv(annotations_file)

        self.img_dir = img_dir
        self.transform = transform 
        self.target_transform = target_transform 

    def __len__(self): 
        return len(self.img_labels) 
    
    def __getitem__(self, idx): 
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        
        # This implicitly handles PNG -- wooo!
        # https://pytorch.org/vision/main/generated/torchvision.io.decode_image.html#torchvision.io.decode_image
        #image = torchvision.io.decode_image(input=img_path, mode=torchvision.io.ImageReadMode.GRAY)
        image = torchvision.io.decode_image(input=img_path)

        label = self.img_labels.iloc[idx, 1]
        if self.transform: 
            image = self.transform(image)  
        if self.target_transform: 
            label = self.target_transform(label) 
        return image, label 
    
def get_data_loader(annotations_file, img_dir, batch_size=5): 
    """
    Retrieve a pytorch-style dataloader 
    """

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    data = ShapeRxDataset(annotations_file, img_dir, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    return loader

def train(loader, model, iterations=2):
    """
    Train the model with the provided dataset
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.train()
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss() 

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(iterations):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(loader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs.to(device)
            labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            # TODO: something about our loss calculation or labels is busted ... we are starting
            # with a really low loss that monotically increases. fine if the network is getting 
            # worse than random guesses, but it shouldn't start out at a low loss reliably
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
    
    return "Training complete!"

def eval(): 
    """
    Evaluate the model on a test set
    """
    pass 

def save_model(model, path):
    """
    Save the model to a file
    """
    torch.save(model, os.path.join(path, "cnn.pt"))
    print(f"Model saved to {path}")

def load_model(path): 
    model = torch.load(os.path.join(path, "cnn.pt"))

    return model