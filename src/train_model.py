import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def train_model(keypoints_path, labels_path, model_output_dir):
    """
    Loads preprocessed data, builds, trains, and saves an LSTM model for pose classification.

    Args:
        keypoints_path (str): Path to the keypoints CSV file.
        labels_path (str): Path to the labels CSV file.
        model_output_dir (str): Directory to save the trained model and label encoder.
    """
    # --- 1. Load Data ---
    print("Loading data...")
    if not os.path.exists(keypoints_path) or not os.path.exists(labels_path):
        print(f"Error: Data files not found. Please run the preprocessing script first.")
        print(f"Expected keypoints at: {keypoints_path}")
        print(f"Expected labels at: {labels_path}")
        return

    X = pd.read_csv(keypoints_path)
    y = pd.read_csv(labels_path)

    print(f"Data loaded successfully. Found {len(X)} frames and {len(y)} labels.")
    print(f"Number of unique asanas found: {y['label'].nunique()}")
    print("Asanas:", y['label'].unique())

    # --- 2. Prepare Data ---
    print("Preparing data for training...")

    # Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y['label'])
    y_categorical = to_categorical(y_encoded)

    # Save the label encoder
    os.makedirs(model_output_dir, exist_ok=True)
    encoder_path = os.path.join(model_output_dir, 'label_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {encoder_path}")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    # Reshape data for LSTM [samples, timesteps, features]
    # We treat each frame as a single timestep for simplicity here.
    # A more advanced approach would be to group frames into sequences.
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    num_classes = y_categorical.shape[1]
    input_shape = (X_train.shape[1], X_train.shape[2])

    # --- 3. Build the Model ---
    print("Building the LSTM model...")
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        LSTM(128, return_sequences=False, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # --- 4. Train the Model ---
    print("\nStarting model training...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # --- 5. Evaluate the Model ---
    print("\nEvaluating model performance...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    # --- 6. Save the Model ---
    model_path = os.path.join(model_output_dir, 'yoga_pose_model.keras')
    model.save(model_path)
    print(f"\nTraining complete. Model saved to {model_path}")


if __name__ == '__main__':
    # --- Configuration ---
    processed_data_path = os.path.join('data', 'processed_keypoints')
    keypoints_file = os.path.join(processed_data_path, 'keypoints_data.csv')
    labels_file = os.path.join(processed_data_path, 'labels.csv')
    model_dir = 'models'

    train_model(keypoints_file, labels_file, model_dir)
