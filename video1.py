import cv2
import os

# Define paths to the input and output directories
input_dir = 'C:\\Users\\Sahil\\Downloads\\test\\input_dir'
output_dir = 'C:\\Users\\Sahil\\Downloads\\test\\frame_video1'

# Define the size of the output frames
width, height = 224, 224

# Loop through all videos in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.mp4'):
        
        # Open the video file
        video = cv2.VideoCapture(os.path.join(input_dir, filename))
        
        # Create a new directory for the output frames
        output_subdir = os.path.join(output_dir, os.path.splitext(filename)[0])
        os.makedirs(output_subdir, exist_ok=True)
        
        # Loop through all frames in the video
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            # Resize the frame
            frame = cv2.resize(frame, (width, height))
            
            # Save the frame to the output directory
            output_filename = os.path.join(output_subdir, f'{frame_count:05d}.jpg')
            cv2.imwrite(output_filename, frame)
            
            frame_count += 1
            
        video.release()