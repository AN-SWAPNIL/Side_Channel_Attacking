import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

# Configuration
DATASET_PATH = "dataset.json"
MODELS_DIR = "saved_models"
BATCH_SIZE = 64
EPOCHS = 50  
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8 
INPUT_SIZE = 1000  
HIDDEN_SIZE = 128

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


class FingerprintClassifier(nn.Module):
    """Basic neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(FingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After two 2x pooling operations
        self.fc_input_size = conv_output_size * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
class ComplexFingerprintClassifier(nn.Module):
    """A more complex neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexFingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After three 2x pooling operations
        self.fc_input_size = conv_output_size * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size*2)
        self.bn4 = nn.BatchNorm1d(hidden_size*2)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x



def train(model, train_loader, test_loader, criterion, optimizer, epochs, model_save_path):
    """Train a PyTorch model and evaluate on the test set.
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        model_save_path: Path to save the best model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(traces)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * traces.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Evaluation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for traces, labels in test_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs = model(traces)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * traces.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_accuracy = correct / total
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_accuracy)
        
        # Print status
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, '
              f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}')
        
        # Save the best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with accuracy: {best_accuracy:.4f}')
    
    return best_accuracy



def evaluate(model, test_loader, website_names):
    """Evaluate a PyTorch model on the test set and show classification report with website names.
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for testing data
        website_names: List of website names for classification report
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for traces, labels in test_loader:
            traces, labels = traces.to(device), labels.to(device)
            outputs = model(traces)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print classification report with website names instead of indices
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=website_names,
        zero_division=1
    ))
    
    return all_preds, all_labels


class WebsiteFingerprintDataset(Dataset):
    """Dataset class for website fingerprint data."""
    def __init__(self, data):
        self.traces = []
        self.labels = []
        
        for item in data:
            # Extract the trace data and website index
            trace = item['trace_data']
            label = item['website_index']
            
            # Ensure trace is the right length - pad or truncate
            if len(trace) < INPUT_SIZE:
                # Pad with zeros if trace is too short
                trace = trace + [0] * (INPUT_SIZE - len(trace))
            elif len(trace) > INPUT_SIZE:
                # Truncate if trace is too long
                trace = trace[:INPUT_SIZE]
                
            self.traces.append(np.array(trace, dtype=np.float32))
            self.labels.append(label)
            
        self.traces = np.array(self.traces)
        self.labels = np.array(self.labels)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return torch.tensor(self.traces[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


def main():
    """ Train and evaluate website fingerprinting models using dataset.json.
    1. Load the dataset from the JSON file
    2. Split the dataset into training and testing sets
    3. Create data loaders for training and testing
    4. Define the models to train
    5. Train and evaluate each model
    6. Print comparison of results
    """
    print(f"Loading dataset from {DATASET_PATH}...")
    
    # Load dataset
    try:
        with open(DATASET_PATH, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} trace samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    # Extract unique websites and their indices
    website_indices = {}
    website_names = []
    for item in data:
        website = item['website']
        index = item['website_index']
        if index not in website_indices:
            website_indices[index] = website
            website_names.append(website)
            
    num_classes = len(website_names)
    print(f"Found {num_classes} unique websites:")
    for i, website in enumerate(website_names):
        print(f"  {i}: {website}")
    
    # Create dataset
    dataset = WebsiteFingerprintDataset(data)
    print(f"Dataset created with {len(dataset)} samples")
    
    # Split dataset into train and test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-TRAIN_SPLIT, random_state=42)
    train_indices, test_indices = next(sss.split(dataset.traces, dataset.labels))
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"Split dataset: {len(train_dataset)} training samples, {len(test_dataset)} testing samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize models
    models = {
        "SimpleModel": FingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes),
        "ComplexModel": ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    }
    
    # Define loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        
        # Define optimizer for this model
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Setup model save path
        model_save_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
        
        # Train model
        best_accuracy = train(model, train_loader, test_loader, criterion, optimizer, EPOCHS, model_save_path)
        
        # Load the best model for evaluation
        model.load_state_dict(torch.load(model_save_path))
        
        # Evaluate model
        print(f"\n=== Evaluating {model_name} ===")
        all_preds, all_labels = evaluate(model, test_loader, website_names)
        
        # Save results
        results[model_name] = {
            "accuracy": best_accuracy,
            "predictions": all_preds,
            "labels": all_labels
        }
        
    # Compare results
    print("\n=== Model Comparison ===")
    for model_name, result in results.items():
        print(f"{model_name}: Test Accuracy = {result['accuracy']:.4f}")
    
    print("\nTraining complete! Models saved in:", MODELS_DIR)


if __name__ == "__main__":
    main()
