#
# 
#
import numpy as np
from stl import mesh
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def extract_features(stl_file):
    model_mesh = mesh.Mesh.from_file(stl_file)
    num_faces = len(model_mesh.vectors)
    vertices = model_mesh.vectors

    def calculate_volume_and_area(mesh_data):
        total_volume = 0
        total_area = 0
        for vector in mesh_data:
            a = vector[1] - vector[0]
            b = vector[2] - vector[0]
            cross_product = np.cross(a, b)
            area = np.linalg.norm(cross_product) / 2
            total_area += area
            volume = np.dot(vector[0], np.cross(a, b)) / 6
            total_volume += volume
        return abs(total_volume), total_area

    volume, area = calculate_volume_and_area(vertices)
    return [num_faces, volume, area]


def prepare_data(stl_folder):
    features = []
    labels = []
    for filename in os.listdir(stl_folder):
        if filename.endswith(".stl"):
            file_path = os.path.join(stl_folder, filename)
            features.append(extract_features(file_path))
            
            # Extract label from filename
            label = filename.split('_')[1].split('.')[0]
            labels.append(label)
    
    X = np.array(features)
    y = np.array(labels)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_svm(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['rbf', 'poly', 'relu']}
    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    return best_model

from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Folder path
stl_folder = "data"


# Splitting data
X_train, X_test, y_train, y_test, scaler = prepare_data(stl_folder)

# Model training
svm_model = train_svm(X_train, y_train)

# Evaluating model
accuracy, report = evaluate_model(svm_model, X_test, y_test)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Testing Data
new_stl_file = "data/002056_vertebraeC5.stl"
new_features = extract_features(new_stl_file)
new_features_scaled = scaler.transform([new_features])
prediction = svm_model.predict(new_features_scaled)
print(f"Prediction for new STL file: {prediction[0]}")





