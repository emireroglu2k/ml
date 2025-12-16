import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import joblib 
import time
import sys

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
BASE_DIR = 'dataset'
train_dir = os.path.join(BASE_DIR, 'preprocessed_train')
val_dir = os.path.join(BASE_DIR, 'validation') 

MODEL_FILENAME = 'svm_resnet_model.pkl'
SCALER_FILENAME = 'scaler_resnet.pkl'

# --- 1. Setup ResNet50 (Feature Extractor) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading ResNet50...")
# Load pre-trained ResNet50
weights = models.ResNet50_Weights.IMAGENET1K_V1
resnet = models.resnet50(weights=weights)

# Remove the last classification layer (fc) to get the feature vector (2048 features)
modules = list(resnet.children())[:-1]
feature_extractor = nn.Sequential(*modules)
feature_extractor.to(device)
feature_extractor.eval() # Set to evaluation mode

print("ResNet50 loaded.")

# Define Image Transforms (Resize -> Tensor -> Normalize)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_resnet_features(img_path):
    """
    Reads image, pushes through ResNet, returns 2048 features.
    """
    try:
        # Load Image (PIL handles formats better for PyTorch)
        img = Image.open(img_path).convert('RGB')
        
        # Preprocess
        img_tensor = preprocess(img)
        # Add batch dimension (1, 3, 224, 224)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = feature_extractor(img_tensor)
        
        # Flatten (1, 2048, 1, 1) -> (2048,)
        return features.flatten().cpu().numpy()
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def process_directory(data_root_dir, data_type="Training"):
    features = []
    labels = []
    
    if not os.path.exists(data_root_dir):
        print(f"Error: Directory not found: {data_root_dir}")
        sys.exit(1)

    classes = sorted(os.listdir(data_root_dir))
    
    for breed_name in classes:
        breed_path = os.path.join(data_root_dir, breed_name)
        if not os.path.isdir(breed_path): continue
            
        print(f"[{data_type}] Processing Breed: {breed_name}")
        
        for filename in os.listdir(breed_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(breed_path, filename)
                feats = get_resnet_features(img_path)
                if feats is not None:
                    features.append(feats)
                    labels.append(breed_name)
                    
    return np.array(features), np.array(labels)

if __name__ == '__main__':
    start_time = time.time()
    
    # --- 1. Extract Features ---
    print("\n--- Phase 1: Feature Extraction (ResNet50) ---")
    X_train, y_train = process_directory(train_dir, "Train")
    X_val, y_val = process_directory(val_dir, "Val")
    
    X_full = np.vstack((X_train, X_val))
    y_full = np.concatenate((y_train, y_val))
    
    print(f"\nExtraction complete in {time.time() - start_time:.2f}s")
    print(f"Total Samples: {X_full.shape[0]}, Dimensions: {X_full.shape[1]}")

    # --- 2. Scaling ---
    print("\n--- Phase 2: Scaling ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    # --- 3. Train SVM ---
    print("\n--- Phase 3: Training SVM ---")
    param_grid = {
        'C': [0.1, 1, 10], 
        'kernel': ['linear', 'rbf'] 
    }
    
    svm = SVC(class_weight='balanced', probability=True, random_state=42)
    grid = GridSearchCV(svm, param_grid, cv=3, verbose=2, n_jobs=-1)
    
    grid.fit(X_scaled, y_full)
    
    print(f"\nBest Train Accuracy: {grid.best_score_:.4f}")
    print(f"Best Params: {grid.best_params_}")
    
    # Save
    joblib.dump(grid.best_estimator_, MODEL_FILENAME)
    joblib.dump(scaler, SCALER_FILENAME)
    print("\nModel saved successfully.")