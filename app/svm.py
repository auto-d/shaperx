#
# 
#

import numpy as np
from PIL import Image

def load_image(path):
    img = Image.open(path).convert('L')  
    
    # Check if image is within specified range
    width, height = img.size
    if not (32 <= width <= 256 and 32 <= height <= 256):
        raise ValueError(f"Image size must be between 32x32 and 256x256. Current size: {width}x{height}")
    img_array = np.array(img)  
    return img_array.flatten()  

def svm_model_train(data, labels):
    
    from sklearn.svm import SVC
    svm_model = SVC()
    svm_model.fit(data, labels)
    return svm_model
    

def svm_model_prediction(svm_model,new_data):

    prediction= svm_model.predict(new_data)
    return prediction





