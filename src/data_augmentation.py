import cv2
import numpy as np
import os
import random

def augment_video(video_path, output_folder, augmentation_functions):
    """
    Applies a list of augmentation functions to a video and saves the augmented videos.

    Args:
        video_path (str): The path to the input video file.
        output_folder (str): The folder to save the augmented videos.
        augmentation_functions (list): A list of augmentation functions to apply.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get original video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    base_filename = os.path.splitext(os.path.basename(video_path))[0]

    for aug_func in augmentation_functions:
        # Create a meaningful name for the augmented file
        # The lambda functions don't have a __name__, so we handle that
        try:
            aug_name = aug_func.__name__
        except AttributeError:
            # For lambda functions, create a generic name
            aug_name = f"random_aug_{random.randint(1000,9999)}"

        output_path = os.path.join(output_folder, f"{base_filename}_{aug_name}.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Reset video to the beginning for each augmentation
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            augmented_frame = aug_func(frame)

            # IMPORTANT: Resize frame back to original dimensions in case an
            # augmentation (like scaling) changed it.
            if augmented_frame.shape[0] != frame_height or augmented_frame.shape[1] != frame_width:
                augmented_frame = cv2.resize(augmented_frame, (frame_width, frame_height))

            out.write(augmented_frame)

        out.release()
        print(f"Saved augmented video: {output_path}")

    cap.release()

def flip_horizontal(frame):
    """Horizontally flips the frame."""
    return cv2.flip(frame, 1)

def rotate(frame, angle):
    """Rotates the frame by a given angle."""
    rows, cols, _ = frame.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(frame, M, (cols, rows))

def scale(frame, scale_factor):
    """Scales the frame by a given factor."""
    rows, cols, _ = frame.shape
    # Note: cv2.resize expects (width, height) which is (cols, rows)
    return cv2.resize(frame, (int(cols * scale_factor), int(rows * scale_factor)), interpolation=cv2.INTER_AREA)

def add_noise(frame):
    """Adds random Gaussian noise to the frame."""
    mean = 0
    var = 10
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, frame.shape).astype('uint8')
    noisy_frame = cv2.add(frame, gaussian)
    return noisy_frame

def change_brightness(frame, value):
    """Changes the brightness of the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)


if __name__ == '__main__':
    # --- Configuration ---
    # Set the root directories based on your project structure
    root_input_folder = os.path.join("data", "raw_videos")
    root_output_folder = os.path.join("data", "augmented_videos")

    # --- List of augmentations to apply ---
    # We use lambda functions to pass random parameters each time
    augmentations = [
        flip_horizontal,
        lambda frame: rotate(frame, angle=random.uniform(-10, 10)),
        lambda frame: scale(frame, scale_factor=random.uniform(0.9, 1.1)),
        add_noise,
        lambda frame: change_brightness(frame, value=random.randint(-40, 40)),
    ]

    # --- Process all videos using the new folder structure ---
    print(f"Starting augmentation from '{root_input_folder}'...")

    # Check if the input directory exists
    if not os.path.isdir(root_input_folder):
        print(f"Error: Input directory not found at '{root_input_folder}'")
        print("Please make sure your 'data/raw_videos' directory is set up correctly.")
    else:
        # os.walk will traverse the directory tree for us
        for dirpath, dirnames, filenames in os.walk(root_input_folder):
            for filename in filenames:
                if filename.lower().endswith((".mp4", ".avi", ".mov")):
                    # Construct the full path to the input video
                    video_path = os.path.join(dirpath, filename)

                    # Create the corresponding output directory structure
                    # 'dirpath' will be like 'data/raw_videos/asana/rating'
                    # We replace 'raw_videos' with 'augmented_videos' to get the output path
                    relative_path = os.path.relpath(dirpath, root_input_folder)
                    output_dir = os.path.join(root_output_folder, relative_path)

                    # Create the output directory if it doesn't exist
                    os.makedirs(output_dir, exist_ok=True)

                    print(f"\nProcessing: {video_path}")
                    augment_video(video_path, output_dir, augmentations)

        print("\nData augmentation complete!")
        print(f"Augmented videos are saved in '{root_output_folder}'")

