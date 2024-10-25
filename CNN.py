import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

"""
This script processes a dataset of signatures to train a Convolutional Neural Network (CNN) for classification. 
It sets parameters such as the output directory, number of epochs, and batch size, then loads training and testing data from .npz files. 
The script filters out invalid labels(which is :-1), reshapes and normalizes the images, and one-hot encodes the labels for training.
It defines a CNN architecture with multiple convolutional and pooling layers, compiles the model using categorical cross-entropy loss, and 
trains the model while saving it as a .keras file for future use.
"""


output_dir = 'E:\\IMPORTED FROM C\\Desktop\\UNIVERSITY\\SEMESTER 7\\GEN_AI\\Assignment1_Q1\\signature_dataset'
epochs = 25
batch_size = 32

data_train = np.load(os.path.join(output_dir, 'train.npz'))
X_train, y_train = data_train['X_train'], data_train['y_train']
data_test = np.load(os.path.join(output_dir, 'test.npz'))
X_test, y_test = data_test['X_test'], data_test['y_test']

X_train_filtered = X_train[y_train != -1]
y_train_filtered = y_train[y_train != -1]
X_test_filtered = X_test[y_test != -1]
y_test_filtered = y_test[y_test != -1]

y_train_filtered = y_train_filtered - 1
y_test_filtered = y_test_filtered - 1

X_train_filtered = X_train_filtered.reshape(-1, 128, 128, 1).astype('float32') / 255.0
X_test_filtered = X_test_filtered.reshape(-1, 128, 128, 1).astype('float32') / 255.0

y_train_one_hot = to_categorical(y_train_filtered, num_classes=185)
y_test_one_hot = to_categorical(y_test_filtered, num_classes=185)

cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(185, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cnn_history = cnn_model.fit(X_train_filtered, y_train_one_hot, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=2)
cnn_model.save('signature_cnn_model.keras')
