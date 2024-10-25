# Signature-Recognition-Using-CNN

## RESEARCH PAPER: 
[Report_Signature_Recognition_using_CNN.pdf](https://github.com/user-attachments/files/17523897/Report_Signature_Recognition_using_CNN.pdf)



This project focuses on signature recognition by developing a program to identify individuals based on their signature images. Utilizing Convolutional Neural Networks (CNNs), the project processes and segments signatures, performs a train-test split, and evaluates the effectiveness of CNN-based feature extraction compared to traditional methods like HOG (Histogram of Oriented Gradients) and SIFT (Scale-Invariant Feature Transform).

# Objectives
Process and segment signature images for each individual.

Perform a train-test split on the dataset.

Implement a CNN model for signature recognition.

Compare CNN-based feature extraction with traditional methods (HOG, SIFT).

Analyze model performance using metrics such as precision, recall, F-measure, and overall accuracy.

Provide visualizations of train and test errors or accuracies.


# Tools and Technologies

Programming Language: Python


# Libraries:

OpenCV: For image processing and feature extraction (HOG, SIFT).

TensorFlow/Keras: For implementing the CNN model.

NumPy: For numerical operations and data manipulation.

pandas: For data handling and manipulation.

Matplotlib/Seaborn: For visualizations and plotting train-test errors or accuracies.


# Getting Started

### Clone the Repository:

git clone https://github.com/yourusername/signature-recognition-cnn.git

cd signature-recognition-cnn


### Install Required Libraries: Ensure you have all necessary libraries installed:

pip install -r requirements.txt


# Dataset Preparation:

Place your signature images in the appropriate directory structure.

The program will process and segment these images into separate folders for each individual.

Running the Program: Execute the main script to process images, train the CNN model, and evaluate its performance:


# Evaluation Metrics


## Precision: 
Measures the accuracy of positive predictions.

## Recall: 
Measures the ability of the model to find all relevant cases.

## F-measure: 
The harmonic mean of precision and recall.

## Overall Accuracy: 
The ratio of correctly predicted instances to the total instances.

## Results and Visualizations
The project includes detailed descriptions and visualizations supporting the analysis of model performance

![Screenshot 2024-09-29 212057](https://github.com/user-attachments/assets/38bb1fec-cd12-4e17-ab0e-4ed7601e5bec)



# Plots of training and testing accuracy and error rates.

Comparison charts illustrating the performance of CNN, HOG, and SIFT methods.
![Screenshot 2024-09-29 204641](https://github.com/user-attachments/assets/dee6df1b-5176-4a46-aaf1-6c4c3abe79ad)
![Screenshot 2024-09-29 204733](https://github.com/user-attachments/assets/9fb814db-348a-4658-926e-985a5dc5336c)


# Contributing
Contributions are welcome! If you have suggestions for improvements or additional features, feel free to submit issues or pull requests.



License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
OpenCV Documentation
TensorFlow Documentation
Keras Documentation
