import cv2
import os

# Define the input and output paths
input_dir = 'C:\\Users\\Sahil\\Downloads\\test\\input_dir'
output_dir = 'C:\\Users\\Sahil\\Downloads\\test\\output_frames'
# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the video file extensions to process
video_extensions = ['.mp4', '.avi', '.mov']

# Loop through all the videos in the input directory
for file_name in os.listdir(input_dir):
    if os.path.splitext(file_name)[1] in video_extensions:
        video_path = os.path.join(input_dir, file_name)

        # Open the video file for reading
        video_capture = cv2.VideoCapture(video_path)

        # Initialize the frame counter
        frame_count = 0

        # Loop through all the frames in the video
        while True:
            # Read the next frame from the video
            ret, frame = video_capture.read()

            # If there are no more frames, exit the loop
            if not ret:
                break

            # Save the current frame as a JPEG image
            output_file_name = os.path.splitext(file_name)[0] + '_' + str(frame_count) + '.jpg'
            output_file_path = os.path.join(output_dir, output_file_name)
            cv2.imwrite(output_file_path, frame)

            # Increment the frame counter
            frame_count += 1

        # Release the video capture object
        video_capture.release()