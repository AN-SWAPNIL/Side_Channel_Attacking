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
DATASET_PATH = "merged_dataset.json"
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
            
            # Forward + backward + optimize
            outputs = model(traces)
            loss = criterion(outputs, labels)
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
                
                # Statistics
                running_loss += loss.item() * traces.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_accuracy = correct / total
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_accuracy)
        
        # Print status
        print(f'Epoch {epoch+1}/{epochs} | ' +
              f'Train Loss: {train_losses[-1]:.4f} | ' +
              f'Train Acc: {train_accuracies[-1]:.4f} | ' +
              f'Test Loss: {test_losses[-1]:.4f} | ' +
              f'Test Acc: {test_accuracies[-1]:.4f}')
        
        # Save the best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with accuracy: {best_accuracy:.4f}")
    
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


def main():
    """Main function to train and evaluate the models for website fingerprinting.
    1. Load the dataset from the JSON file
    2. Split the dataset into training and testing sets
    3. Create data loaders for training and testing
    4. Define the models to train
    5. Train and evaluate each model
    6. Print comparison of results
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. Load the dataset from JSON file
    print("Loading dataset...")
    with open(DATASET_PATH, 'r') as f:
        dataset_raw = json.load(f)
    
    # Process the dataset based on the JSON format with multiple entries per website
    websites_data = {}
    
    for entry in dataset_raw:
        website_url = entry["website"]
        website_index = entry["website_index"]
        trace_data = entry["trace_data"]
        
        # Add website to dictionary if it doesn't exist
        if website_url not in websites_data:
            websites_data[website_url] = {
                "index": website_index,
                "traces": []
            }
        
        # Add trace data to the website
        websites_data[website_url]["traces"].append(trace_data)
    
    # Extract website names and ensure balanced dataset
    website_names = list(websites_data.keys())
    num_classes = len(website_names)
    print(f"Found {num_classes} websites:")
    
    # Print summary of each website's data
    for i, website in enumerate(website_names):
        print(f"  [{websites_data[website]['index']}] {website}: {len(websites_data[website]['traces'])} items")
    
    # Determine the minimum number of traces across all websites
    min_traces = min(len(websites_data[website]["traces"]) for website in website_names)
    
    # Optionally balance the dataset - comment this out if you want to use all available data
    # min_traces = min(len(websites_data[website]["traces"]) for website in website_names)
    # print(f"Balancing dataset to {min_traces} traces per website...")
    
    # Convert data to suitable format for PyTorch
    all_traces = []
    all_labels = []
    
    for website in website_names:
        traces = websites_data[website]["traces"]
        website_index = websites_data[website]["index"]
        
        # Use all traces (or limit to min_traces for balanced dataset)
        # traces = traces[:min_traces]  # Uncomment this line to balance the dataset
        
        for trace in traces:
            # Ensure all traces have the same length by padding or truncating
            if len(trace) > INPUT_SIZE:
                trace = trace[:INPUT_SIZE]  # Truncate
            elif len(trace) < INPUT_SIZE:
                trace = trace + [0] * (INPUT_SIZE - len(trace))  # Pad with zeros
                
            all_traces.append(trace)
            all_labels.append(website_index)  # Use the actual website_index from the dataset
    
    # Convert to numpy arrays
    all_traces = np.array(all_traces, dtype=np.float32)
    all_labels = np.array(all_labels)
    
    print(f"Dataset shape: {all_traces.shape}, Labels shape: {all_labels.shape}")
    
    # 2. Normalize the data (Min-Max scaling)
    print("Preprocessing data...")
    trace_min = all_traces.min()
    trace_max = all_traces.max()
    all_traces = (all_traces - trace_min) / (trace_max - trace_min + 1e-8)
    
    # 3. Split the dataset into training and testing sets using stratified sampling
    # to maintain class balance
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-TRAIN_SPLIT, random_state=42)
    train_idx, test_idx = next(sss.split(all_traces, all_labels))
    
    # Create PyTorch Dataset
    class TraceDataset(Dataset):
        def __init__(self, traces, labels):
            self.traces = torch.FloatTensor(traces)
            self.labels = torch.LongTensor(labels)
            
        def __len__(self):
            return len(self.labels)
            
        def __getitem__(self, idx):
            return self.traces[idx], self.labels[idx]
    
    # Create full dataset
    full_dataset = TraceDataset(all_traces, all_labels)
    
    # Split into train and test datasets
    train_dataset = Subset(full_dataset, train_idx)
    test_dataset = Subset(full_dataset, test_idx)
    
    print(f"Training set size: {len(train_dataset)}, Testing set size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. Define the models to train
    print("Setting up models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define models
    simple_model = FingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    complex_model = ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    
    # Define loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    simple_optimizer = optim.Adam(simple_model.parameters(), lr=LEARNING_RATE)
    complex_optimizer = optim.Adam(complex_model.parameters(), lr=LEARNING_RATE)
    
    # 5. Train and evaluate each model
    print("\n==== Training Simple Model ====")
    simple_model_path = os.path.join(MODELS_DIR, "SimpleModel.pth")
    simple_accuracy = train(
        simple_model, 
        train_loader, 
        test_loader, 
        criterion, 
        simple_optimizer, 
        EPOCHS, 
        simple_model_path
    )
    
    print("\n==== Training Complex Model ====")
    complex_model_path = os.path.join(MODELS_DIR, "ComplexModel.pth")
    complex_accuracy = train(
        complex_model, 
        train_loader, 
        test_loader, 
        criterion, 
        complex_optimizer, 
        EPOCHS, 
        complex_model_path
    )
    
    # 6. Load the best models and evaluate
    print("\n==== Evaluating Best Models ====")
    
    # Map indices to website names for the report
    website_mapping = {}
    for website in website_names:
        website_mapping[websites_data[website]["index"]] = website
        
    report_names = [website_mapping[i] for i in sorted(website_mapping.keys())]
    
    # Load and evaluate simple model
    print("\nSimple Model Evaluation:")
    simple_model = FingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    simple_model.load_state_dict(torch.load(simple_model_path))
    simple_preds, simple_labels = evaluate(simple_model, test_loader, report_names)
    
    # Load and evaluate complex model
    print("\nComplex Model Evaluation:")
    complex_model = ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    complex_model.load_state_dict(torch.load(complex_model_path))
    complex_preds, complex_labels = evaluate(complex_model, test_loader, report_names)
    
    # 7. Print comparison of results
    print("\n==== Model Comparison ====")
    print(f"Simple Model Accuracy: {simple_accuracy:.4f}")
    print(f"Complex Model Accuracy: {complex_accuracy:.4f}")
    
    if simple_accuracy > complex_accuracy:
        print("The Simple Model performed better!")
    else:
        print("The Complex Model performed better!")
    
    print("\nTraining and evaluation complete!")


if __name__ == "__main__":
    main()
