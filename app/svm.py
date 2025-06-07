
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import glob

def load_image(path):
    img = Image.open(path).convert('L')  
    
    # Check if image is within specified range
    width, height = img.size
    if not (32 <= width <= 500 and 32 <= height <= 500):
        raise ValueError(f"Image size must be between 32x32 and 500x500. Current size: {width}x{height}")
    img_array = np.array(img)  
    return img_array.flatten()  

def svm_model_train(data, labels):
    
    
    svm_model = SVC()
    svm_model.fit(data, labels)
    return svm_model
    

def svm_model_prediction(svm_model,new_data):

    prediction= svm_model.predict(new_data)
    return prediction

def load_dataset(data_dir, categories):
   
    data = []
    labels = []
    filenames = []
    
    for category_idx, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category path {category_path} does not exist. Skipping.")
            continue
            
        print(f"Loading {category} images...")
        image_paths = glob.glob(os.path.join(category_path, "*.jpg")) + \
                     glob.glob(os.path.join(category_path, "*.png"))
        
        for image_path in image_paths:
            try:
  
                img_features = load_image(image_path)
                data.append(img_features)
                labels.append(category_idx)
                filenames.append(os.path.basename(image_path))
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                
    return np.array(data), np.array(labels), filenames


def train_validate_model(data_dir, categories, test_size=0.2, random_state=0, save_model_path=None):

    # Load dataset
    X, y, filenames = load_dataset(data_dir, categories)
    
    if len(X) == 0:
        print("No valid images found. Please double-checkcheck data directory.")
        return None
    
    print(f"Loaded {len(X)} images with shape: {X[0].shape}")
    

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Train/validate SVM 
    print("Training SVM model...")
    model = svm_model_train(X_train, y_train)
    print("Validating model...")
    y_pred = svm_model_prediction(model, X_val)
    
    # Calculate best fit 
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=categories))
    
    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        model_data = {
            'model': model,
            'scaler': scaler,
            'categories': categories
        }
        joblib.dump(model_data, save_model_path)
        print(f"Model saved to {save_model_path}")
    
    return model, scaler

def predict_on_new_images(model, scaler, categories, image_paths):

    predictions = []
    for image_path in image_paths:
        try:
         
            img_features = load_image(image_path)
            img_features_scaled = scaler.transform([img_features])
            
            # Predictions
            prediction = svm_model_prediction(model, img_features_scaled)[0]
            predicted_category = categories[prediction]
            
            predictions.append({
                'image': image_path,
                'prediction': predicted_category,
                'prediction_idx': prediction
            })
            
            # Display image and prediction
            img = plt.imread(image_path)
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
            plt.title(f"THe predicted class is: {predicted_category}")
            plt.axis('off')
            plt.show()
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return predictions

def eval(): 
    """
    Evaluate the model on a test set
    """
    pass 

def save_model(model:SVC, path):
    """
    Save the model to a file
    """    
    filename = os.path.join(path, 'svm.pkl')
    with open(filename, 'wb') as f: 
        pickle.dump(model, f)
    
    print(f"Model saved to {path}")

def load_model(path): 
    """
    Load our naive model off disk
    """
    model = None

    filename = os.path.join(path, 'svm.pkl')
    with open(filename, 'r') as f: 
        model = pickle.load(f) 
    
    if type(model) != SVC: 
        raise ValueError(f"Unexpected type {type(model)} found in {filename}")

    return model

