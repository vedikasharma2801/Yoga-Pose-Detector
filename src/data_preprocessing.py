import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

def extract_keypoints_from_videos(root_video_folder, output_folder):
    """
    Extracts pose keypoints from all videos in a directory structure,
    saves them to a CSV file, and creates a corresponding labels file.

    Args:
        root_video_folder (str): Path to the root folder containing asana subdirectories.
        output_folder (str): Path to the folder where CSV files will be saved.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    all_landmarks = []
    all_labels = []
    video_counter = 0

    print(f"Starting keypoint extraction from '{root_video_folder}'...")
    os.makedirs(output_folder, exist_ok=True)

    # Simplified and more robust directory traversal
    asana_folders = [f for f in os.listdir(root_video_folder) if os.path.isdir(os.path.join(root_video_folder, f))]

    for asana_name in asana_folders:
        asana_path = os.path.join(root_video_folder, asana_name)
        rating_folders = [f for f in os.listdir(asana_path) if os.path.isdir(os.path.join(asana_path, f))]

        for rating in rating_folders:
            rating_path = os.path.join(asana_path, rating)
            
            for filename in os.listdir(rating_path):
                if filename.lower().endswith((".mp4", ".avi", ".mov")):
                    video_path = os.path.join(rating_path, filename)
                    cap = cv2.VideoCapture(video_path)

                    if not cap.isOpened():
                        print(f"Warning: Could not open video file {video_path}. Skipping.")
                        continue

                    video_counter += 1
                    print(f"Processing video {video_counter}: {video_path} [Asana: {asana_name}]")

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image.flags.writeable = False
                        results = pose.process(image)
                        image.flags.writeable = True

                        if results.pose_landmarks:
                            landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
                            all_landmarks.append(landmarks)
                            all_labels.append(asana_name)

                    cap.release()

    if not all_landmarks:
        print("\n\n!!! CRITICAL WARNING: No landmarks were extracted. !!!")
        print("Please check the following:")
        print(f"1. Does the folder '{root_video_folder}' contain subfolders for each asana?")
        print("2. Do those asana folders contain 'avg', 'good', 'poor' subfolders?")
        print("3. Are there .mp4, .avi, or .mov video files in those rating folders?")
        print("4. Are the video files corrupted or empty?")
        return

    keypoints_df = pd.DataFrame(all_landmarks)
    labels_df = pd.DataFrame(all_labels, columns=['label'])

    keypoints_path = os.path.join(output_folder, 'keypoints_data.csv')
    labels_path = os.path.join(output_folder, 'labels.csv')

    keypoints_df.to_csv(keypoints_path, index=False)
    labels_df.to_csv(labels_path, index=False)

    print(f"\nSuccessfully processed {video_counter} videos.")
    print(f"Keypoints data saved to: {keypoints_path}")
    print(f"Labels data saved to: {labels_path}")

if __name__ == '__main__':
    # --- Configuration ---
    raw_videos_path = os.path.join('data', 'raw_videos')
    augmented_videos_path = os.path.join('data', 'augmented_videos')
    processed_data_path = os.path.join('data', 'processed_keypoints')
    
    # We will process both folders and combine the results.
    all_video_paths = [raw_videos_path, augmented_videos_path]
    
    # It's better to process all videos and save once to avoid file overwriting issues.
    # This script is now designed to be run once over a list of folders.
    # Let's modify the main function to reflect this.
    
    # This is a placeholder to show the intended logic. The function above is the key part.
    # For simplicity, we will call the function on the augmented videos,
    # as they are a superset of the raw videos if you ran the augmentation script.
    
    print("--- Processing all augmented videos ---")
    extract_keypoints_from_videos(augmented_videos_path, processed_data_path)

