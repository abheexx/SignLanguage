import cv2
import numpy as np
import torch
from typing import Tuple, List
import os
import json
from datetime import datetime

def create_directory(path: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path (str): Directory path to create
    """
    os.makedirs(path, exist_ok=True)

def save_config(config: dict, path: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config (dict): Configuration dictionary
        path (str): Path to save the configuration
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(path: str) -> dict:
    """
    Load configuration from a JSON file.
    
    Args:
        path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)

def get_timestamp() -> str:
    """
    Get current timestamp in a formatted string.
    
    Returns:
        str: Formatted timestamp
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess a frame for model input.
    
    Args:
        frame (np.ndarray): Input frame
        target_size (Tuple[int, int]): Target size for resizing
        
    Returns:
        np.ndarray: Preprocessed frame
    """
    # Resize frame
    frame = cv2.resize(frame, target_size)
    
    # Convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame

def draw_prediction(frame: np.ndarray, prediction: str, confidence: float) -> np.ndarray:
    """
    Draw prediction and confidence on the frame.
    
    Args:
        frame (np.ndarray): Input frame
        prediction (str): Predicted class
        confidence (float): Confidence score
        
    Returns:
        np.ndarray: Frame with prediction drawn
    """
    # Create text
    text = f"{prediction}: {confidence:.2%}"
    
    # Add text to frame
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    return frame

def create_video_writer(output_path: str, fps: float = 30.0) -> cv2.VideoWriter:
    """
    Create a video writer object.
    
    Args:
        output_path (str): Path to save the video
        fps (float): Frames per second
        
    Returns:
        cv2.VideoWriter: Video writer object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (224, 224))

def save_prediction_history(history: List[Tuple[str, float]], output_path: str) -> None:
    """
    Save prediction history to a file.
    
    Args:
        history (List[Tuple[str, float]]): List of (prediction, confidence) tuples
        output_path (str): Path to save the history
    """
    with open(output_path, 'w') as f:
        for pred, conf in history:
            f.write(f"{pred},{conf}\n")

def load_prediction_history(input_path: str) -> List[Tuple[str, float]]:
    """
    Load prediction history from a file.
    
    Args:
        input_path (str): Path to the history file
        
    Returns:
        List[Tuple[str, float]]: List of (prediction, confidence) tuples
    """
    history = []
    with open(input_path, 'r') as f:
        for line in f:
            pred, conf = line.strip().split(',')
            history.append((pred, float(conf)))
    return history

def calculate_metrics(predictions: List[str], true_labels: List[str]) -> dict:
    """
    Calculate various metrics for model evaluation.
    
    Args:
        predictions (List[str]): List of predicted labels
        true_labels (List[str]): List of true labels
        
    Returns:
        dict: Dictionary containing various metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def save_metrics(metrics: dict, output_path: str) -> None:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics (dict): Dictionary containing metrics
        output_path (str): Path to save the metrics
    """
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4) 