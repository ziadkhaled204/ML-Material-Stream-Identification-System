from skimage.feature import hog
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

input_dir = "./dataset"
categories = ["cardboard","glass","metal","paper","plastic","trash"]

data = []
labels = []
img2vec = Img2Vec()

def augment_pil(img):
    imgs = [img]
    imgs.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    imgs.append(img.rotate(15))
    imgs.append(img.rotate(-15))
    
    return imgs

def extract_features(img):
    img = cv2.resize(img, (64,64))

    # --- HOG ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
    gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    feature_vector=True
    )

    # --- Color Histogram ---
    hist = cv2.calcHist([img], [0,1,2], None, [16,16,16], [0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()

    return np.hstack((hog_features, hist))

for label_index, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("Skipped corrupted image:", img_path)
            continue

        augmented_imgs = augment_pil(img)

        for aug in augmented_imgs:
            aug_cv = cv2.cvtColor(np.array(aug), cv2.COLOR_RGB2BGR)
            features = extract_features(aug_cv)
            data.append(features)
            labels.append(label_index)
        

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_classifier = SVC(kernel='rbf', gamma='scale', C=10)
svm_classifier.fit(X_train, y_train)

#knn_classifer = KNeighborsClassifier(n_neighbors=7, weights='distance')
#knn_classifer.fit(X_train, y_train)


y_pred = svm_classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
