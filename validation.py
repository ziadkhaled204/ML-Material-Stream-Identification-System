import os
import cv2
import pickle
import numpy as np
from PIL import Image
from img2vec_pytorch import Img2Vec

with open("models/svm_model.pkl", "rb") as f:
    svm_classifier = pickle.load(f)

with open("models/knn_model.pkl", "rb") as f:
    knn_classifier = pickle.load(f)

with open("models/scaler_model.pkl", "rb") as f:
    scaler = pickle.load(f)

categories = ["cardboard","glass","metal","paper","plastic","trash"]
threshold_svm = 5.2
KNN_PROB_THRESHOLD = 0.8      # confidence among neighbors
KNN_DIST_THRESHOLD = 30     # distance to training data


img2vec = Img2Vec()

# Helper function to get features
def extract_features(pil_img):
    features = img2vec.get_vec(pil_img)
    return features.flatten()

#to validate the predictions
val_dir = "./val"

print("\nValidation Predictions:\n" + "-"*30)

for file in os.listdir(val_dir):
    img_path = os.path.join(val_dir, file)

    try:
        img = Image.open(img_path).convert("RGB")
    except:
        print(f"{file}: could not load")
        continue

    # Extract & scale features
    features = extract_features(img)
    features_scaled = scaler.transform([features])

    # SVM prediction
    svm_scores = svm_classifier.decision_function(features_scaled)
    max_svm_score = np.max(svm_scores)

    if max_svm_score < threshold_svm:
        svm_label = "Unknown"
    else:
        svm_index = svm_classifier.predict(features_scaled)[0]
        svm_label = categories[svm_index]

    # KNN prediction
    knn_proba = knn_classifier.predict_proba(features_scaled)
    max_knn_prob = np.max(knn_proba)
    
    distances, _ = knn_classifier.kneighbors(features_scaled)
    avg_distance = np.mean(distances)

    if max_knn_prob < KNN_PROB_THRESHOLD or avg_distance > KNN_DIST_THRESHOLD :
        knn_label = "Unknown"
    else:
        knn_index = knn_classifier.predict(features_scaled)[0]
        knn_label = categories[knn_index]
        
        

    print(f"{file}  -->  SVM: {svm_label},  KNN: {knn_label}")
    