import cv2
import numpy as np
from sklearn.svm import SVC
import joblib as joblib

from freature_extraction import extract_features


# Load the trained SVM classifier
classifier = joblib.load('C:\\Users\\Sahil\\Downloads\\test\\SVM classifier\\svm_classifier.joblib')

# Define the video capture device
cap = cv2.VideoCapture(0)

# Define the Region of Interest (ROI) for hand detection
top, right, bottom, left = 10, 350, 225, 590

# Define the background subtractor for hand segmentation
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # Flip the frame horizontally for a more natural mirror effect
    frame = cv2.flip(frame, 1)
    
    # Crop the frame to the ROI for hand detection
    roi = frame[top:bottom, right:left]
    
    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply background subtraction to segment the hand
    fg_mask = bg_subtractor.apply(blur, learningRate=0)
    
    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the largest contour (i.e., the hand) on the original frame
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(roi, [largest_contour], 0, (0, 255, 0), 2)
        
        # Extract features from the hand contour (e.g., Hu moments, Fourier descriptors, etc.)
        features = extract_features(largest_contour)
        
        # Classify the hand gesture using the trained SVM classifier
        label = classifier.predict(features.reshape(1, -1))[0]
        
        # Display the predicted label on the original frame
        cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the original frame with the hand contour and the predicted label
    cv2.imshow('Sign Language Detection', frame)
    
    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
