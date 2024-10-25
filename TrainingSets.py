import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Path for dataset and output directory
data_dir = 'processed_signatures1'
output_dir = 'signature_dataset'

def preprocess_and_split(output_dir, test_size=0.2):
    """
    This function processes the dataset of signatures by loading images from the specified 
    directory, extracting their corresponding labels (student IDs), and performing a train-test 
    split. It reads each image in grayscale and stores them in a list along with their labels. 
    After encoding the labels, it splits the dataset into training and testing sets based on 
    the given test size ratio. The resulting sets are then saved as .npz files in the specified 
    output directory, ensuring they can be easily loaded for future model training or evaluation.

    Parameters:
    - output_dir (str): The directory where the processed data will be saved.
    - test_size (float): The proportion of the dataset to include in the test split (default is 0.2).
    """
    X, y = [], []
    label_encoder = LabelEncoder()
    
    for student_dir in os.listdir(data_dir):
        student_path = os.path.join(data_dir, student_dir)
        if os.path.isdir(student_path):
            for img_file in os.listdir(student_path):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(student_path, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    X.append(img)
                    y.append(student_dir)

    X = np.array(X)
    y = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np.savez(os.path.join(output_dir, 'train.npz'), X_train=X_train, y_train=y_train)
    np.savez(os.path.join(output_dir, 'test.npz'), X_test=X_test, y_test=y_test)
    print(f"Data split completed, saved to {output_dir}")

preprocess_and_split(output_dir)
