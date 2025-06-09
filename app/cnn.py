#
# Adapated from PyTorch image classifier tutorial - https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#

from torchsummary import summary
import os
import math 
import pandas as pd
import torch
import torchvision 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from mesh_dataset import class_map

class Net(nn.Module):
    """
    Basic convolutional neural network derived from the pytorch tutorial 
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    """

    @classmethod
    def filter_size(cls, w, f, p=0, s=1):
        """
        Helper to sanity check filter output dimensions at runtime
        output_size = [ (W - F + 2P) / S ] + 1
        """
        return ((w - f + 2 * p) / s ) + 1

    def __init__(self, w=32):
        super().__init__()
        
        self.dyn_convs = nn.ModuleList(self.build_scaling_layers(w))
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(class_map))

    def build_scaling_layers(self, w): 
        """
        Build scaling layers to handle input size dynamically
        """
        convs = []
        
        # There are only a handful of resolutions we allow at the moment, but this logic
        # will scale to arbitrary 32x2^x widths. Anyway, here are the iterations this 
        # loop produces for our supported image sizes
        # 32 -> 0 scale adjustments:
        # 64 -> 1 scale adjustments to:
        #  32.0
        # 128 -> 2 scale adjustments to:
        #  -64.0
        #  -32.0
        # 256 -> 3 scale adjustments to 
        #  -128.0
        #  -64.0
        #  -32.0
        scale = math.log2(w/32)
        if scale.is_integer(): 
            for i in range(int(scale)):
                print(f"Creating dynamic filter for scaling @ {scale} ({w} pixel width)...")
                convs.append(nn.Conv2d(in_channels=3,out_channels=3,kernel_size=8,stride=2, padding=3))
                w = Net.filter_size(w=w,f=8,s=2,p=3)
            
        return convs

    def forward(self, x):
        # 0. Convolution shim to learn features only relevant at larger image scales
        for conv in self.dyn_convs: 
            x = conv(x)
            x = F.relu(x)

        # 1. Convolution + pooling
        #
        # Each filter here (conv_2d out_channels) is a unique opportunity to 
        # learn a useful pattern in the input. The filter count used is based on 
        # the pytorch example, but if we had the compute we would want to do a 
        # search over these values (as with most other network parameters here)!

        # in:  3 x 32^2
        # conv @ 6, 5: 6 x 28^2
        # pool @ 2x2:  6 x 14^2
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # 2. Convolution + pooling
        # input:       6 x 14^2 
        # conv @ 16,5: 16 x 10^2
        # pool @ 2x2:  16 x 5^2

        # out_dims = 14 - 5 + 1 = 10
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
    
def get_data_loader(annotations_file, img_dir, batch_size=5, shuffle=True): 
    """
    Retrieve a pytorch-style dataloader 
    """

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    data = ShapeRxDataset(annotations_file, img_dir, transform=transform)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    
    return loader

def train(loader, model, loss_interval=20, epochs=2, lr=0.01, momentum=0.9):
    """
    Train the model with the provided dataset
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_loss = []

    model.train()
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(loader):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # collect metrics
            running_loss += loss.item()

            if (i % loss_interval) == (loss_interval - 1): 
                train_loss.append(running_loss / loss_interval)
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / loss_interval:.3f}")
                running_loss = 0 

    return train_loss 

from skorch import NeuralNetClassifier
from skorch.helper import SkorchDoctor

def train_debug(loader, y, epochs, lr, momentum): 

    net = NeuralNetClassifier(
        module=Net, 
        criterion=nn.CrossEntropyLoss, 
        optimizer=optim.SGD, 
        max_epochs=epochs, 
        lr=lr, 
        optimizer__momentum=momentum, 
        verbose=0)
    
    dr = SkorchDoctor(net) 

    dr.fit(loader.dataset, y=y)

    return dr

def predict(loader, model): 
    """
    Run a dataset through the (hopefully trained) model and return outputs
    """

    preds = []

    # Reduce the memory required for a forward pass by disabling the 
    # automatic gradient computation (i.e. commit to not call backward()
    # after this pass)
    with torch.no_grad(): 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device) 
        model.eval() 

        # Compute the logits for every class and grab the class
        # TODO: switch this to top-k? 
        for i, data in enumerate(loader): 
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model(inputs) 
            
            preds.append(outputs.index[max(outputs)])

    return preds

def save_model(model, path):
    """
    Save the model to a file
    """
    filename = os.path.join(path, "cnn.pt")
    torch.save(model, filename)
    print(f"Model saved to {filename}")

    return filename

def load_model(path): 
    model = torch.load(os.path.join(path, "cnn.pt"), weights_only=False)

    return model