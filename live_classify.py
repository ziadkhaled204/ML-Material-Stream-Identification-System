import os
import cv2
import pickle
import numpy as np
from PIL import Image
from img2vec_pytorch import Img2Vec

# -----------------------------
# Load saved models and scaler
# -----------------------------
with open("models/svm_model.pkl", "rb") as f:
    svm_classifier = pickle.load(f)

with open("models/knn_model.pkl", "rb") as f:
    knn_classifier = pickle.load(f)

with open("models/scaler_model.pkl", "rb") as f:
    scaler = pickle.load(f)

categories = ["cardboard","glass","metal","paper","plastic","trash"]
threshold_svm = 5.2
KNN_PROB_THRESHOLD = 0.8      # confidence among neighbors
KNN_DIST_THRESHOLD = 30     # distance to training data (tune once)

img2vec = Img2Vec()

# Helper function to get features
def extract_features(pil_img):
    features = img2vec.get_vec(pil_img)
    return features.flatten()

# Camera
cap = cv2.VideoCapture(0)

# Define ROI box coordinates (x, y, width, height)
roi_x, roi_y, roi_w, roi_h = 200, 100, 200, 200  # adjust as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw rectangle on screen
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (0, 255, 0), 2)

    # Crop the ROI
    roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    # Convert to PIL for img2vec
    pil_img = Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))

    # Extract features and scale
    features = extract_features(pil_img)
    features_scaled = scaler.transform([features])

    # SVM
    svm_scores = svm_classifier.decision_function(features_scaled)
    max_svm_score = np.max(svm_scores)
    if max_svm_score < threshold_svm:
        svm_label = "Unknown"
    else:
        svm_index = svm_classifier.predict(features_scaled)[0]
        svm_label = categories[svm_index]

    # KNN 
    knn_proba = knn_classifier.predict_proba(features_scaled)
    max_knn_prob = np.max(knn_proba)
    
    distances, _ = knn_classifier.kneighbors(features_scaled)
    avg_distance = np.mean(distances)

    if max_knn_prob < KNN_PROB_THRESHOLD or avg_distance > KNN_DIST_THRESHOLD :
        knn_label = "Unknown"
    else:
        knn_index = knn_classifier.predict(features_scaled)[0]
        knn_label = categories[knn_index]
        

    # Display labels
    cv2.putText(frame, f"SVM: {svm_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"KNN: {knn_label}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()