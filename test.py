import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import cv2
from datetime import datetime
import fiftyone as fo
import fiftyone.zoo as foz
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    DATABASE_NAME = "Voxel51/GMNCSA24-FO"
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    NUM_CLASSES = None  # Will be set dynamically
    FALL_LABEL = "Falling (BW)"  # The label that indicates a fall
    FRAME_SAMPLE_RATE = 5  # Sample every Nth frame from videos
    MODEL_SAVE_PATH = "models/fall_detector.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    THRESHOLD = 0.75  # Confidence threshold for fall detection alert

# Custom dataset class for video frames
class ActivityDataset(Dataset):
    def __init__(self, samples, label_map, transform=None):
        self.samples = samples
        self.label_map = label_map
        self.transform = transform
        self.frames = []
        self.labels = []
        
        # Extract frames and labels
        logger.info("Preparing dataset...")
        self._extract_frames_and_labels()
        
    def _extract_frames_and_labels(self):
        for sample in tqdm(self.samples):
            filepath = sample.filepath
            event_label = sample.events.label
            label_idx = self.label_map[event_label]
            
            # Handle both image and video files
            if filepath.endswith(('.mp4', '.avi', '.mov')):
                cap = cv2.VideoCapture(filepath)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Sample frames at regular intervals
                for i in range(0, frame_count, Config.FRAME_SAMPLE_RATE):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frames.append(frame)
                        self.labels.append(label_idx)
                
                cap.release()
            else:  # Handle image files
                frame = cv2.imread(filepath)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame)
                self.labels.append(label_idx)
        
        logger.info(f"Dataset prepared with {len(self.frames)} frames")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame, label

# Model definition (using ResNet50 as base)
class ActivityDetector(nn.Module):
    def __init__(self, num_classes):
        super(ActivityDetector, self).__init__()
        self.model = models.resnet50(pretrained=True)
        
        # Replace the final layer for our classification task
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Function to connect to FiftyOne and load the dataset
def load_dataset():
    try:
        logger.info(f"Connecting to FiftyOne database: {Config.DATABASE_NAME}")
        dataset = fo.load_dataset(Config.DATABASE_NAME)
        logger.info(f"Successfully loaded dataset with {len(dataset)} samples")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)

# Function to prepare the data for training
def prepare_data(dataset):
    # Get unique activity labels
    unique_labels = set()
    for sample in dataset:
        if hasattr(sample, 'events') and hasattr(sample.events, 'label'):
            unique_labels.add(sample.events.label)
    
    # Create a mapping from label string to index
    label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    Config.NUM_CLASSES = len(label_map)
    
    logger.info(f"Found {Config.NUM_CLASSES} unique activity classes")
    logger.info(f"Label mapping: {label_map}")
    
    # Save the label mapping for inference
    with open("label_map.json", "w") as f:
        json.dump(label_map, f)
    
    # Split the dataset into train and validation
    train_samples, val_samples = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Only resize and normalize for validation
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset instances
    train_dataset = ActivityDataset(train_samples, label_map, transform=train_transform)
    val_dataset = ActivityDataset(val_samples, label_map, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, label_map

# Training function
def train_model(train_loader, val_loader, label_map):
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    
    # Initialize model
    model = ActivityDetector(Config.NUM_CLASSES).to(Config.DEVICE)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    fall_detection_accuracies = []
    
    logger.info(f"Starting training for {Config.NUM_EPOCHS} epochs on {Config.DEVICE}")
    
    for epoch in range(Config.NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]"):
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        fall_correct = 0
        fall_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Val]"):
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                
                # Calculate fall detection accuracy
                fall_idx = label_map[Config.FALL_LABEL]
                predictions = torch.argmax(outputs, dim=1)
                
                # Find samples with fall label
                fall_mask = (labels == fall_idx)
                fall_total += fall_mask.sum().item()
                
                # Count correct fall detections
                if fall_total > 0:
                    fall_correct += ((predictions == fall_idx) & fall_mask).sum().item()
        
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate fall detection accuracy
        fall_accuracy = fall_correct / max(1, fall_total)  # Avoid division by zero
        fall_detection_accuracies.append(fall_accuracy)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        logger.info(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Fall Detection Accuracy: {fall_accuracy:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'fall_accuracy': fall_accuracy,
            }, Config.MODEL_SAVE_PATH)
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(fall_detection_accuracies, label='Fall Detection Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Fall Detection Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    
    logger.info("Training completed successfully")
    logger.info(f"Best model saved to {Config.MODEL_SAVE_PATH}")

# Fall detection and alert system
class FallDetectionSystem:
    def __init__(self, model_path, label_map_path):
        # Load label map
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        
        # Reverse label map for inference
        self.idx_to_label = {idx: label for label, idx in self.label_map.items()}
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActivityDetector(len(self.label_map)).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Set up transform for inference
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Fall detection system initialized")
    
    def process_frame(self, frame):
        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(frame_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item()
            
        activity = self.idx_to_label[prediction]
        
        # Check if fall detected
        is_fall = (activity == Config.FALL_LABEL) and (confidence >= Config.THRESHOLD)
        
        return activity, confidence, is_fall
    
    def trigger_alert(self, frame, activity, confidence):
        # Save the frame that triggered the alert
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert_dir = "fall_alerts"
        os.makedirs(alert_dir, exist_ok=True)
        
        alert_path = os.path.join(alert_dir, f"fall_alert_{timestamp}.jpg")
        cv2.imwrite(alert_path, frame)
        
        # Log the alert
        logger.warning(f"⚠️ FALL DETECTED with confidence {confidence:.2f}! Alert saved to {alert_path}")
        
        # In a real system, this could send an SMS, email, or trigger other alert systems
        # For demonstration, we'll just print to console and save the frame
        print("\n" + "*" * 50)
        print(f"⚠️ EMERGENCY: FALL DETECTED at {timestamp}")
        print(f"Activity: {activity}, Confidence: {confidence:.2f}")
        print(f"Alert image saved to: {alert_path}")
        print("*" * 50 + "\n")
        
        # You would integrate with external alert systems here
        # Example: send_sms_alert(), send_email_alert(), etc.

# Function to test the model on a video stream
def test_fall_detection(video_path=None):
    # Initialize fall detection system
    fall_detector = FallDetectionSystem(
        model_path=Config.MODEL_SAVE_PATH,
        label_map_path="label_map.json"
    )
    
    # Open video capture (camera or file)
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)  # Use default camera
    
    if not cap.isOpened():
        logger.error("Error: Could not open video source")
        return
    
    # Get video properties for display
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info("Starting fall detection. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        activity, confidence, is_fall = fall_detector.process_frame(frame)
        
        # Display result on frame
        status_color = (0, 0, 255) if is_fall else (0, 255, 0)  # Red for fall, green otherwise
        cv2.putText(frame, f"Activity: {activity}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Display the frame
        cv2.imshow('Fall Detection System', frame)
        
        # Trigger alert if fall detected
        if is_fall:
            fall_detector.trigger_alert(frame, activity, confidence)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def main():
    logger.info("Starting activity and fall detection training pipeline")
    
    # Load dataset
    dataset = load_dataset()
    
    # Prepare data for training
    train_loader, val_loader, label_map = prepare_data(dataset)
    
    # Train the model
    train_model(train_loader, val_loader, label_map)
    
    # Test the model on a sample video if available
    test_videos = [sample.filepath for sample in dataset 
                  if sample.filepath.endswith(('.mp4', '.avi', '.mov'))]
    
    if test_videos:
        test_fall_detection(test_videos[0])
    else:
        logger.info("No test videos found in dataset. Run with a webcam using:")
        logger.info("    python test_fall_detection.py")

if __name__ == "__main__":
    main()