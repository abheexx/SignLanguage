import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import mediapipe as mp
from typing import List, Tuple, Dict
import argparse

class SignLanguageDataset(Dataset):
    def __init__(self, data_dir: str, sequence_length: int = 30, transform=None):
        """
        Initialize the Sign Language Dataset.
        
        Args:
            data_dir (str): Directory containing the dataset
            sequence_length (int): Number of frames in each sequence
            transform: Optional transform to be applied on frames
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """Load all video samples and their labels."""
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for video_name in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_name)
                self.samples.append((video_path, self.class_to_idx[class_name]))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        frames = self._load_video_frames(video_path)
        return frames, label
    
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load and preprocess video frames."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            
            if self.transform:
                frame = self.transform(frame)
            
            frames.append(frame)
        
        cap.release()
        
        # Pad sequence if necessary
        while len(frames) < self.sequence_length:
            frames.append(frames[-1])
        
        return torch.stack(frames)

class HandDetector:
    def __init__(self):
        """Initialize MediaPipe hand detector."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect_hands(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect hands in the frame and return processed frame with hand landmarks.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Tuple[np.ndarray, List]: Processed frame and hand landmarks
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on frame
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        
        return frame, results.multi_hand_landmarks if results.multi_hand_landmarks else []

def get_data_loader(data_dir: str, batch_size: int = 32, sequence_length: int = 30) -> DataLoader:
    """
    Create and return a DataLoader for the sign language dataset.
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size for the DataLoader
        sequence_length (int): Number of frames in each sequence
        
    Returns:
        DataLoader: DataLoader instance
    """
    dataset = SignLanguageDataset(data_dir, sequence_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

def preprocess_dataset(data_dir: str, output_dir: str):
    """
    Preprocess the entire dataset and save processed videos.
    
    Args:
        data_dir (str): Directory containing raw dataset
        output_dir (str): Directory to save processed videos
    """
    detector = HandDetector()
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        for video_name in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_name)
            output_path = os.path.join(output_class_dir, video_name)
            
            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (224, 224))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame = cv2.resize(frame, (224, 224))
                frame, _ = detector.detect_hands(frame)
                out.write(frame)
            
            cap.release()
            out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Dataset Preprocessing")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing raw dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed videos")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the dataset")
    
    args = parser.parse_args()
    
    if args.preprocess:
        preprocess_dataset(args.data_dir, args.output_dir) 