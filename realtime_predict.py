import cv2
import torch
import numpy as np
from typing import List, Tuple, Optional
from data_loader import HandDetector
from model import CNNLSTMModel
from torchvision import transforms

class SignLanguagePredictor:
    def __init__(self, model_path: str, class_names: List[str], sequence_length: int = 30):
        """
        Initialize the Sign Language Predictor.
        
        Args:
            model_path (str): Path to the trained model weights
            class_names (List[str]): List of class names
            sequence_length (int): Number of frames in each sequence
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNNLSTMModel(num_classes=len(class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = class_names
        self.sequence_length = sequence_length
        self.hand_detector = HandDetector()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.frame_buffer = []
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single frame for prediction.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            torch.Tensor: Preprocessed frame
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = self.transform(frame)
        return frame
    
    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Predict sign language gesture from a frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Tuple[str, float]: Predicted class and confidence
        """
        # Detect hands and process frame
        frame, landmarks = self.hand_detector.detect_hands(frame)
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Add to frame buffer
        self.frame_buffer.append(processed_frame)
        if len(self.frame_buffer) > self.sequence_length:
            self.frame_buffer.pop(0)
        
        # If we don't have enough frames yet, return None
        if len(self.frame_buffer) < self.sequence_length:
            return None, 0.0
        
        # Prepare input tensor
        input_tensor = torch.stack(self.frame_buffer).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            predicted_class = self.class_names[prediction.item()]
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def process_video_stream(self, cap: cv2.VideoCapture) -> Tuple[np.ndarray, Optional[str], float]:
        """
        Process a single frame from video stream.
        
        Args:
            cap (cv2.VideoCapture): Video capture object
            
        Returns:
            Tuple[np.ndarray, Optional[str], float]: Processed frame, prediction, and confidence
        """
        ret, frame = cap.read()
        if not ret:
            return None, None, 0.0
        
        # Make prediction
        prediction, confidence = self.predict(frame)
        
        # Add prediction text to frame
        if prediction:
            text = f"{prediction}: {confidence:.2f}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, prediction, confidence

def main():
    """Test the predictor with webcam feed."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Sign Language Recognition")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights")
    parser.add_argument("--class_names", type=str, nargs="+", required=True, help="List of class names")
    
    args = parser.parse_args()
    
    predictor = SignLanguagePredictor(args.model_path, args.class_names)
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            frame, prediction, confidence = predictor.process_video_stream(cap)
            if frame is None:
                break
            
            cv2.imshow("Sign Language Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 