import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.feature import hog
import cv2  # OpenCV for SIFT
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score

output_dir = r'E:\IMPORTED FROM C\Desktop\UNIVERSITY\SEMESTER 7\GEN_AI\I212478_MuhammaadTayyabSohail_Ass1\Assignment1_Q1\signature_dataset'

"""
This script evaluates a pre-trained Convolutional Neural Network (CNN) model on a test dataset of signatures,
loading the model and test data from specified directories. It filters out invalid labels, reshapes, and normalizes 
the test images before making predictions with the CNN. The script then generates a classification report and confusion
matrix to assess the model's performance. Additionally, it implements feature extraction techniques using Histogram of
Oriented Gradients (HOG) and Scale-Invariant Feature Transform (SIFT), training a Multi-Layer Perceptron (MLP) classifier
on the extracted features. The performance of the MLP models is evaluated similarly, with classification reports and 
confusion matrices visualized for analysis.
"""

cnn_model = tf.keras.models.load_model('signature_cnn_model.keras')

data_test = np.load(os.path.join(output_dir, 'test.npz'))
X_test, y_test = data_test['X_test'], data_test['y_test']

X_test_filtered = X_test[y_test != -1]
y_test_filtered = y_test[y_test != -1]
y_test_filtered = y_test_filtered - 1
X_test_filtered = X_test_filtered.reshape(-1, 128, 128, 1).astype('float32') / 255.0

y_pred = cnn_model.predict(X_test_filtered)
y_pred_classes = np.argmax(y_pred, axis=1)

unique_classes = np.unique(y_test_filtered)
print(classification_report(y_test_filtered, y_pred_classes, target_names=[str(i) for i in unique_classes], labels=unique_classes))

cm = confusion_matrix(y_test_filtered, y_pred_classes)
class_names = ['Class A', 'Class B', 'Class C', 'Class D']

def plot_confusion_matrix(cm, class_names, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=.5, linecolor='gray', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

num_classes_to_show = 4
cm_reduced = cm[:num_classes_to_show, :num_classes_to_show]
plot_confusion_matrix(cm_reduced, class_names[:num_classes_to_show], 'Reduced Confusion Matrix')

f1 = f1_score(y_test_filtered, y_pred_classes, average='macro')
precision = precision_score(y_test_filtered, y_pred_classes, average='macro')
accuracy = accuracy_score(y_test_filtered, y_pred_classes)
recall = recall_score(y_test_filtered, y_pred_classes, average='macro')

print(f"F1 Score: {f1:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")

def extract_hog_features(X):
    hog_features = []
    for img in X:
        img = img.reshape(128, 128)
        features = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)

X_train_hog = extract_hog_features(X_test_filtered)

mlp_model_hog = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_model_hog.fit(X_train_hog, y_test_filtered)

y_pred_hog = mlp_model_hog.predict(X_train_hog)
print("\nClassification Report for MLP with HOG features:")
print(classification_report(y_test_filtered, y_pred_hog, target_names=[str(i) for i in unique_classes], labels=unique_classes))

cm_hog = confusion_matrix(y_test_filtered, y_pred_hog)
plot_confusion_matrix(cm_hog[:num_classes_to_show, :num_classes_to_show], class_names[:num_classes_to_show], 'Reduced Confusion Matrix for MLP Model with HOG Features')

def extract_sift_features(X):
    sift_features = []
    sift = cv2.SIFT_create()
    for img in X:
        img = img.reshape(128, 128)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            sift_features.append(descriptors.flatten())
        else:
            sift_features.append(np.zeros((128,)))
    return np.array(sift_features)

X_train_sift = extract_sift_features(X_test_filtered)

mlp_model_sift = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp_model_sift.fit(X_train_sift, y_test_filtered)

y_pred_sift = mlp_model_sift.predict(X_train_sift)
print("\nClassification Report for MLP with SIFT features:")
print(classification_report(y_test_filtered, y_pred_sift, target_names=[str(i) for i in unique_classes], labels=unique_classes))

cm_sift = confusion_matrix(y_test_filtered, y_pred_sift)
plot_confusion_matrix(cm_sift[:num_classes_to_show, :num_classes_to_show], class_names[:num_classes_to_show], 'Reduced Confusion Matrix for MLP Model with SIFT Features')
