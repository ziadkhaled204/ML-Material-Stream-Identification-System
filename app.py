import os
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from img2vec_pytorch import Img2Vec
from sklearn.model_selection import GridSearchCV
from PIL import ImageEnhance


input_dir = "./dataset"
categories = ["cardboard","glass","metal","paper","plastic","trash"]

# Initialize Img2Vec
img2vec = Img2Vec()

features_file = "features.npy"
labels_file = "labels.npy"

if os.path.exists(features_file) and os.path.exists(labels_file):
    print("Loading cached features...")
    data = np.load(features_file)
    labels = np.load(labels_file)
else:
    data = []
    labels = []

    def augment_pil(img):
        imgs = [img]
        
        #zoom
        width, height = img.size
        left = int(0.05*width)
        top = int(0.05*height)
        right = int(0.95*width)
        bottom = int(0.95*height)
        img.crop((left, top, right, bottom)).resize((width, height))

        # Flips
        imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))

        # Rotations
        imgs.append(img.rotate(15))
        imgs.append(img.rotate(-15))

        # Brightness/Contrast
        enhancer = ImageEnhance.Brightness(img)
        imgs.append(enhancer.enhance(1.2))
        imgs.append(enhancer.enhance(0.8))

        enhancer = ImageEnhance.Contrast(img)
        imgs.append(enhancer.enhance(1.2))
        imgs.append(enhancer.enhance(0.8))

        return imgs

    def extract_features(pil_img):
        features = img2vec.get_vec(pil_img)
        return features.flatten()
    
    # Load dataset
    for label_index, category in enumerate(categories):
        category_path = os.path.join(input_dir, category)
        for file in os.listdir(category_path):
            img_path = os.path.join(category_path, file)
            try:
                img = Image.open(img_path).convert("RGB")
                for aug in augment_pil(img):
                    feats = extract_features(aug)
                    data.append(feats)           # add each augmented image
                    labels.append(label_index)   # same label
            except Exception as e:
                print("Skipped corrupted image:", img_path)
                continue

            
                
    np.save(features_file, data)
    np.save(labels_file, labels)            

    data = np.array(data)
    labels = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize classifiers
svm_classifier = SVC(kernel= "rbf", C= 10, gamma="scale")
knn_classifier = KNeighborsClassifier(n_neighbors=3,metric="euclidean",weights="distance")

svm_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train,y_train)
# Test accuracy


y_pred_svm_train = svm_classifier.predict(X_train)
y_pred_knn_train = knn_classifier.predict(X_train)

y_pred_svm = svm_classifier.predict(X_test)
y_pred_knn = knn_classifier.predict(X_test)

print("SVM train Accuracy:", accuracy_score(y_train, y_pred_svm_train))
print("KNN train Accuracy:", accuracy_score(y_train, y_pred_knn_train))

print("SVM test Accuracy:", accuracy_score(y_test, y_pred_svm))
print("KNN test Accuracy:", accuracy_score(y_test, y_pred_knn))


#save models

with open("models/svm_model.pkl", "wb") as f:
    pickle.dump(svm_classifier,f)

with open("models/knn_model.pkl", "wb") as f:
    pickle.dump(knn_classifier,f)

with open("models/scaler_model.pkl" , "wb")as f:
    pickle.dump(scaler,f)
