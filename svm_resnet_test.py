import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import joblib 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BASE_DIR = 'dataset'
test_dir = os.path.join(BASE_DIR, 'test') 
MODEL_FILENAME = 'svm_resnet_model.pkl'
SCALER_FILENAME = 'scaler_resnet.pkl'
RESULTS_FILENAME = 'svm_resnet_test_results.txt'

# --- Rebuild Feature Extractor ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Loading ResNet50...")

# Load standard ResNet50
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.to(device)
feature_extractor.eval()

# Transforms (Must match training)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_resnet_features(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = feature_extractor(img_tensor)
        return features.flatten().cpu().numpy()
    except:
        return None

if __name__ == '__main__':
    if not os.path.exists(MODEL_FILENAME):
        print("Run svm_train_resnet.py first!")
        exit()
        
    print("Loading SVM and Scaler...")
    svm = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)

    print("Processing Test Set...")
    X_list = []
    y_list = []
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        exit()

    for breed in os.listdir(test_dir):
        path = os.path.join(test_dir, breed)
        if not os.path.isdir(path): continue
        
        print(f"Testing Breed: {breed}")
        for f in os.listdir(path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                feats = get_resnet_features(os.path.join(path, f))
                if feats is not None:
                    X_list.append(feats)
                    y_list.append(breed)

    if len(X_list) == 0:
        print("Error: No images found in test folder!")
        exit()

    X_raw = np.array(X_list)
    y_test = np.array(y_list)
    
    # Scale -> Predict
    X_scaled = scaler.transform(X_raw)
    y_pred = svm.predict(X_scaled)
    
    # --- REPORT GENERATION ---
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=svm.classes_)
    
    # Create the output string
    output_text = []
    output_text.append("="*60)
    output_text.append(f"SVM + ResNet50 Test Results")
    output_text.append("="*60)
    output_text.append(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    output_text.append("\n--- Classification Report ---")
    output_text.append(report)
    
    output_text.append("\n--- Confusion Matrix ---")
    output_text.append(f"Labels: {list(svm.classes_)}")
    output_text.append(str(cm))
    
    final_output = "\n".join(output_text)

    # 1. Print to Terminal
    print(final_output)
    
    # 2. Save to Text File
    with open(RESULTS_FILENAME, "w", encoding='utf-8') as f:
        f.write(final_output)
        
    print(f"\nSUCCESS: Results saved to {RESULTS_FILENAME}")