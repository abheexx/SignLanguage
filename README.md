# Real-Time Sign Language Recognition System 🎯

A state-of-the-art American Sign Language (ASL) recognition system that uses deep learning to interpret sign language gestures in real-time through webcam feed.

## 🏆 Recognition
This project was selected as the **Best Project among 50+ entries** in the AI/ML showcase, winning the grand prize for its innovative approach and practical implementation.

## 🚀 Features
- Real-time ASL gesture recognition using webcam feed
- Deep learning architecture combining CNN and LSTM
- Live prediction display with confidence scores
- User-friendly web interface
- Support for multiple ASL signs
- Robust preprocessing pipeline

## 🛠️ Tech Stack
- Python 3.8+
- PyTorch
- OpenCV
- Streamlit
- NumPy
- MediaPipe (for hand tracking)
- scikit-learn

## 📋 Prerequisites
```bash
pip install -r requirements.txt
```

## 🎮 Usage
1. Start the web interface:
```bash
streamlit run app.py
```

2. Allow webcam access when prompted
3. Perform ASL signs in front of the camera
4. View real-time predictions

## 🎯 Model Architecture
- **CNN**: Extracts spatial features from video frames
- **LSTM**: Captures temporal patterns across frame sequences
- Combined architecture for robust gesture recognition

## 📊 Training
1. Download the ASL dataset
2. Preprocess the data:
```bash
python data_loader.py --preprocess
```

3. Train the model:
```bash
python train.py --epochs 50 --batch_size 32
```

## 📁 Project Structure
```
├── app.py              # Streamlit web interface
├── model.py            # CNN-LSTM model architecture
├── data_loader.py      # Data preprocessing and loading
├── realtime_predict.py # Real-time prediction logic
├── utils.py           # Utility functions
├── train.py           # Training script
└── requirements.txt    # Project dependencies
```


## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
