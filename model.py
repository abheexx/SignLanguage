import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes, sequence_length=30, hidden_size=128, num_layers=2):
        """
        Initialize the CNN-LSTM model for sign language recognition.
        
        Args:
            num_classes (int): Number of sign language classes to predict
            sequence_length (int): Number of frames in each sequence
            hidden_size (int): Number of features in the LSTM hidden state
            num_layers (int): Number of recurrent layers in LSTM
        """
        super(CNNLSTMModel, self).__init__()
        
        # CNN for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Calculate the size of CNN output
        self.cnn_output_size = 256 * 4 * 4  # Adjust based on your input size
        
        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, channels, height, width)
        
        Returns:
            torch.Tensor: Output predictions
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape for CNN
        c_in = x.view(batch_size * seq_len, c, h, w)
        
        # CNN forward pass
        c_out = self.cnn(c_in)
        c_out = c_out.view(batch_size, seq_len, -1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(c_out)
        
        # Use the last time step's output
        last_output = lstm_out[:, -1, :]
        
        # Final classification
        output = self.fc(last_output)
        
        return output

def get_model(num_classes, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Create and return the model instance.
    
    Args:
        num_classes (int): Number of sign language classes
        device (str): Device to run the model on
    
    Returns:
        CNNLSTMModel: Model instance
    """
    model = CNNLSTMModel(num_classes=num_classes)
    return model.to(device) 