import os
import cv2
import numpy as np
from skimage import feature

# Define paths to the input and output directories
input_dir = 'C:\\Users\\Sahil\\Downloads\\test\\frame_video1'
output_dir = 'C:\\Users\\Sahil\\Downloads\\test\\hog_features'

# Define the feature extractor function
def extract_features(image):
    # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    images_gray = np.zeros(image.shape[:-1], dtype=image.dtype)
    for i, img in enumerate(image):
        images_gray[i] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features
    hog = feature.hog(images_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    
    return hog

# Loop through all video directories in the input directory
for video_dirname in os.listdir(input_dir):
    video_dir = os.path.join(input_dir, video_dirname)
    if os.path.isdir(video_dir):
        
        # Create a new directory for the output features
        output_subdir = os.path.join(output_dir, video_dirname)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Loop through all image files in the video directory
        for filename in os.listdir(video_dir):
            if filename.endswith('.jpg'):
                
                # Load the image file
                image = cv2.imread(os.path.join(video_dir, filename))
                # coeffs = np.array([0.114, 0.587, 0.229])
                # images_gray = (image.astype(np.float) * coeffs).sum(axis=-1, keepdims=True)
                # images_gray = images_gray.astype(image.dtype)
                # print("image", image)
                # print("image_gray", images_gray)
                
                # Extract features from the image
                features = extract_features(image)
                # print("features", features)
                
                # Save the features to the output directory
                output_filename = os.path.join(output_subdir, os.path.splitext(filename)[0] + '.npy')
                np.save(output_filename, features)
