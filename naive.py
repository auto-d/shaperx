import imageio.v3 as iio
import ipympl
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

def histogram(image_path): 
    """
    Generate a histogram of the provided image

    With help from https://datacarpentry.github.io/image-processing/05-creating-histograms.html
    """
    image = iio.imread(image_path, mode="L")
    ski.util.img_as_float(image)
    histogram, bin_edges = np.histogram(image, bins=256, range=(0,1))

    return histogram 

def build_baseline(): 
    """
    Develop a mean histogram for each provided class
    """
    raise NotImplementedError

def classify(image): 
    """
    Using our histogram baselines, attempt to classify an image
    """
    raise NotImplementedError