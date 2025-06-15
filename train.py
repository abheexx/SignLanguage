import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNNLSTMModel
from data_loader import get_data_loader
import argparse
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Get predictions
        pred = output.argmax(dim=1)
        predictions.extend(pred.cpu().numpy())
        true_labels.extend(target.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    return total_loss / len(train_loader), accuracy

def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        tuple: (Average validation loss, accuracy)
    """
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            
            # Get predictions
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    return total_loss / len(val_loader), accuracy

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train Sign Language Recognition Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save model and plots")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--sequence_length", type=int, default=30, help="Number of frames in each sequence")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get class names
    class_names = sorted(os.listdir(args.data_dir))
    num_classes = len(class_names)
    
    # Create model
    model = CNNLSTMModel(num_classes=num_classes)
    model = model.to(device)
    
    # Create data loaders
    train_loader = get_data_loader(
        os.path.join(args.data_dir, "train"),
        batch_size=args.batch_size,
        sequence_length=args.sequence_length
    )
    val_loader = get_data_loader(
        os.path.join(args.data_dir, "val"),
        batch_size=args.batch_size,
        sequence_length=args.sequence_length
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_accuracy = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_accuracy = validate(
            model, val_loader, criterion, device
        )
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(args.output_dir, "sign_language_model.pth"))
            print("Saved best model!")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "training_curves.png"))
    plt.close()
    
    # Plot final confusion matrix
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    plot_confusion_matrix(
        all_labels,
        all_predictions,
        class_names,
        os.path.join(args.output_dir, "confusion_matrix.png")
    )

if __name__ == "__main__":
    main() 