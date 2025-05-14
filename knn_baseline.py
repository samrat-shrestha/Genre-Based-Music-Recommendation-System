import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load
from tqdm import tqdm  # For progress bar

# function to extract features for DTW
def extract_features(file_path, segment_length=1.0):
    #Extract MFCCs for segments of audio for DTW comparison
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Get segments for more efficient processing
        # instead of the whole song to save computation
        segments = []
        hop_length = int(segment_length * sr)
        
        # Use a few segments from different parts of the song
        # Start, middle, and end sections
        segment_indices = [0, len(y)//2, max(0, len(y)-hop_length)]
        
        for start_idx in segment_indices:
            if start_idx + hop_length <= len(y):
                segment = y[start_idx:start_idx+hop_length]
                # Extract MFCCs for this segment
                mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
                segments.append(mfccs.T)  # Transpose to get time as first dimension
        
        return segments
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# DTW distance function
def dtw_distance(x, y):
    #Calculate DTW distance between two sequences
    distance, _ = fastdtw(x, y, dist=euclidean)
    return distance

# KNN with DTW for classification
class DTW_KNN:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def predict(self, X_test):
        predictions = []
        
        for test_segments in tqdm(X_test, desc="Predicting with DTW"):
            # Calculate distances to all training samples
            distances = []
            
            for train_segments, train_label in zip(self.X_train, self.y_train):
                # For each test segment, find closest training segment
                segment_distances = []
                
                for test_segment in test_segments:
                    min_dist = float('inf')
                    
                    for train_segment in train_segments:
                        # Calculate DTW distance
                        dist = dtw_distance(test_segment, train_segment)
                        min_dist = min(min_dist, dist)
                    
                    segment_distances.append(min_dist)
                
                # Average the distances from all segments
                distances.append(np.mean(segment_distances))
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.n_neighbors]
            
            # Get most common class among neighbors
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)

# Function to load dataset
def load_dataset(data_dir, classes, max_files=None):
    #Load audio files and extract features for DTW-KNN training
    features = []
    labels = []
    
    for i, genre in enumerate(classes):
        genre_dir = os.path.join(data_dir, genre)
        print(f"Processing {genre} files...")
        
        files = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]
        if max_files:
            files = files[:max_files]  # Limit number of files per genre for faster development
        
        for filename in tqdm(files, desc=f"Extracting features for {genre}"):
            file_path = os.path.join(genre_dir, filename)
            # Extract features from the file
            extracted_segments = extract_features(file_path)
            
            if extracted_segments and len(extracted_segments) > 0:
                features.append(extracted_segments)
                labels.append(i)
    
    return features, np.array(labels)

if __name__ == "__main__":
    # same genres as in CNN model
    data_dir = "genres_original"
    classes = ['hiphop', 'metal', 'rock']
    
    # Extract features from dataset
    max_files_per_genre = 100
    X, y = load_dataset(data_dir, classes, max_files=max_files_per_genre)
    print(f"Features extracted for {len(X)} songs, with {len(y)} labels")
    
    # Split into training and testing sets(same ratio as in CNN)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Try different values of k
    k_values = [1, 3, 5]
    best_k = 0
    best_accuracy = 0
    
    for k in k_values:
        print(f"\nTesting with k={k}")
        knn_dtw = DTW_KNN(n_neighbors=k)
        knn_dtw.fit(X_train, y_train)
        accuracy = knn_dtw.score(X_test, y_test)
        print(f"K={k}, Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    
    # Train final model with best k
    print(f"\nTraining final model with k={best_k}")
    knn_dtw = DTW_KNN(n_neighbors=best_k)
    knn_dtw.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = knn_dtw.predict(X_test)
    
    # Print classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for DTW-KNN Genre Classification')
    plt.show()
    
    # # Create visualization for accuracy across different k values
    # plt.figure(figsize=(10, 6))
    # plt.plot(k_values, [knn_dtw.score(X_test, y_test) for k in k_values], 
    #          marker='o', linestyle='-', color='blue', label='Test Accuracy')
    # plt.xlabel('Number of Neighbors (k)')
    # plt.ylabel('Accuracy')
    # plt.title('KNN Performance Across Different k Values')
    # plt.xticks(k_values)
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
    # output best result
    print(f"\nDTW-KNN Results with k={best_k}:")
    print(f"Accuracy: {best_accuracy:.4f}")