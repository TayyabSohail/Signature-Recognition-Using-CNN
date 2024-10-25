import cv2
import os
import numpy as np

def preprocess_and_extract_signatures(image_path, output_dir, num_rows=12, num_pics_per_row=4):
    """
    This function takes an image path and output directory as input, processes the image to extract signatures, 
    and saves them into organized folders. It reads the image in grayscale, applies binary thresholding to isolate 
    the signatures, and uses morphological operations to clean the image. The cleaned image is divided into 
    sections based on the specified number of rows and pictures per row. Each extracted signature is cropped, 
    resized to a uniform size (128x128 pixels), and saved in a dedicated directory for each student based on 
    their ID. 

    Parameters:
    - image_path (str): The path to the image file containing signatures.
    - output_dir (str): The directory where the extracted signatures will be saved.
    - num_rows (int): The number of rows to divide the image into (default is 12).
    - num_pics_per_row (int): The number of pictures (signatures) per row (default is 4).
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}. Please check the file path.")
        return

    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    clean_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    img_height, img_width = clean_image.shape[:2]
    row_height = img_height // num_rows
    pic_width = img_width // num_pics_per_row

    for row in range(num_rows):
        student_id = (image_index * num_rows) + (row + 1)
        student_dir = os.path.join(output_dir, f'student_{student_id}')
        
        if not os.path.exists(student_dir):
            os.makedirs(student_dir)
        
        for pic in range(num_pics_per_row):
            x_start = pic * pic_width
            y_start = row * row_height
            x_end = (pic + 1) * pic_width
            y_end = (row + 1) * row_height
            
            signature = clean_image[y_start:y_end, x_start:x_end]
            contours, _ = cv2.findContours(signature, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                signature_cropped = signature[y:y+h, x:x+w]

                if signature_cropped.size == 0:
                    print(f"No valid signature found for student {student_id}, picture {pic + 1}.")
                    continue

                signature_resized = cv2.resize(signature_cropped, (128, 128))
                signature_path = os.path.join(student_dir, f'signature_{student_id}_{pic + 1}.png')
                cv2.imwrite(signature_path, signature_resized)
                print(f'Saved {signature_path}')

data_dir = r'E:\IMPORTED FROM C\Desktop\UNIVERSITY\SEMESTER 7\GEN_AI\Assignment1_Q1\Data'
output_dir = r'processed_signatures1'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_index = 0
for filename in os.listdir(data_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(data_dir, filename)
        print(f'Processing {image_path}...')
        preprocess_and_extract_signatures(image_path, output_dir)
        image_index += 1

for extra_id in range(image_index + 1, 19):
    student_dir = os.path.join(output_dir, f'student_{extra_id}')
    if not os.path.exists(student_dir):
        os.makedirs(student_dir)
    print(f'Created additional folder: {student_dir}')
