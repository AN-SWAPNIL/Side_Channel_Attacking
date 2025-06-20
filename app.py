from flask import Flask, send_from_directory, request, jsonify
# additional imports
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI errors
import matplotlib.pyplot as plt
import numpy as np
import io
import os
from datetime import datetime
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

# Path to the saved models
MODELS_DIR = "./saved_models_merged"
# Define classes (websites)
WEBSITES = [
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com", 
    "https://prothomalo.com",
    "https://www.chaldal.com",
    "https://www.dhakatribune.com"
]

# Define FingerprintClassifier and ComplexFingerprintClassifier classes for model loading
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

# Load the trained models
def load_models():
    device = torch.device("cpu")  # Use CPU for prediction
    
    print(f"Looking for models in directory: {MODELS_DIR}")
    print(f"Current working directory: {os.getcwd()}")
    
    # List files in the models directory
    try:
        if os.path.exists(MODELS_DIR):
            print(f"Contents of {MODELS_DIR}: {os.listdir(MODELS_DIR)}")
        else:
            print(f"Models directory {MODELS_DIR} does not exist")
    except Exception as e:
        print(f"Error listing model directory: {e}")
    
    # Load Simple Model
    input_size = 1000  # Use the same input size as training
    hidden_size = 128
    num_classes = len(WEBSITES)
    
    print(f"Attempting to load models with num_classes = {num_classes}")
    print(f"Website classes: {WEBSITES}")
    
    simple_model = FingerprintClassifier(input_size, hidden_size, num_classes)
    simple_model_path = os.path.join(MODELS_DIR, "SimpleModel.pth") 
    if os.path.exists(simple_model_path):
        try:
            # Load the model state dict
            state_dict = torch.load(simple_model_path, map_location=device)
            
            # Check for size mismatch in the final layer
            if 'fc2.weight' in state_dict and state_dict['fc2.weight'].size(0) != num_classes:
                trained_classes = state_dict['fc2.weight'].size(0)
                print(f"Model was trained with {trained_classes} classes, but we're using {num_classes} classes")
                
                # If the model was trained with a different number of classes,
                # we need to adapt our model to match
                if trained_classes != num_classes:
                    # Recreate the model with the correct number of classes
                    simple_model = FingerprintClassifier(input_size, hidden_size, trained_classes)
            
            # Load the state dict
            simple_model.load_state_dict(state_dict)
            simple_model.eval()
            print("SimpleModel loaded successfully")
        except Exception as e:
            print(f"Error loading SimpleModel: {e}")
            simple_model = None
    else:
        simple_model = None
        print(f"SimpleModel not found at {simple_model_path}")
    
    # Load Complex Model
    complex_model = ComplexFingerprintClassifier(input_size, hidden_size, num_classes)
    complex_model_path = os.path.join(MODELS_DIR, "ComplexModel.pth")
    if os.path.exists(complex_model_path):
        try:
            # Load the model state dict
            state_dict = torch.load(complex_model_path, map_location=device)
            
            # Check for size mismatch in the final layer
            if 'fc3.weight' in state_dict and state_dict['fc3.weight'].size(0) != num_classes:
                trained_classes = state_dict['fc3.weight'].size(0)
                print(f"ComplexModel was trained with {trained_classes} classes, but we're using {num_classes} classes")
                
                # If the model was trained with a different number of classes,
                # we need to adapt our model to match
                if trained_classes != num_classes:
                    # Recreate the model with the correct number of classes
                    complex_model = ComplexFingerprintClassifier(input_size, hidden_size, trained_classes)
            
            # Load the state dict
            complex_model.load_state_dict(state_dict)
            complex_model.eval()
            print("ComplexModel loaded successfully")
        except Exception as e:
            print(f"Error loading ComplexModel: {e}")
            complex_model = None
    else:
        complex_model = None
        print(f"ComplexModel not found at {complex_model_path}")
    
    return simple_model, complex_model

# Load models at startup
simple_model, complex_model = load_models()

app = Flask(__name__)

# Create images and json directories if they don't exist
os.makedirs('images', exist_ok=True)

# For Task 2, we'll use arrays for storage instead of a database
stored_traces = []  # To store trace JSON data
stored_heatmaps = []  # To store heatmap images
trace_ids = 0  # Counter to generate unique IDs

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

@app.route('/json/<path:filename>')
def serve_json(filename):
    return send_from_directory('json', filename)

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/collect_trace', methods=['POST'])
def collect_trace():
    """ 
    Implement the collect_trace endpoint to receive trace data from the frontend and generate a heatmap.
    1. Receive trace data directly from the frontend as JSON
    2. Generate a heatmap using matplotlib
    3. Store the heatmap and trace data in Python arrays
    4. Save the heatmap image to the images directory
    5. Print the first few values of the trace data to the console
    6. Use the trained model to predict the website
    7. Return the heatmap image, statistics, and predictions to the frontend
    """
    try:
        global trace_ids, stored_traces, stored_heatmaps, simple_model, complex_model
        
        data = request.get_json()
        trace_data = data.get('trace_data')
        timestamp = data.get('timestamp')
        
        if not trace_data:
            return jsonify({'error': 'No trace data provided'}), 400
        
        print(f"Received trace data directly from frontend")
        # Print the first few values of the trace data to the console
        print("First few values of trace data:")
        print(json.dumps(trace_data[:5], indent=2))
        
        # Generate heatmap
        trace_array = np.array(trace_data)
        print(f"Original trace data shape: {trace_array.shape}")
        
        # For the heatmap, we want to show the sweep counts per time interval
        # This creates a single horizontal strip where each vertical line is one time interval
        if len(trace_array.shape) == 2:
            # Sum across all cache lines for each time interval
            interval_counts = np.sum(trace_array, axis=1)
        else:
            # If it's already 1D, use as is
            interval_counts = trace_array
        
        # Calculate statistics
        flat_data = trace_array.flatten()
        min_val = int(np.min(flat_data))
        max_val = int(np.max(flat_data))
        
        # Reshape to create a single row for visualization
        heatmap_data = interval_counts.reshape(1, -1)
        print(f"Heatmap data shape: {heatmap_data.shape}")
        
        plt.figure(figsize=(15, 2))  # Very wide and short - single horizontal strip
        # Use hot colormap so max=red, min=yellow
        plt.imshow(heatmap_data, cmap='hot', interpolation='nearest', aspect='auto')
        plt.yticks([])
        plt.tight_layout()
        
        # Generate current timestamp
        timestamp = datetime.now().isoformat()
        filename = f"heatmap_{timestamp.replace(':', '-').replace('.', '-')}.png"
        filepath = os.path.join('images', filename)
        
        # Save image to file only
        plt.savefig(filepath, format='png', dpi=100, bbox_inches='tight')
        print(f"Saved heatmap image to: {filepath}")
        plt.close()
        
        # Generate current timestamp if not provided
        if not timestamp:
            timestamp = datetime.now().isoformat()
        
        # Create URL path for the frontend (relative to static serving)
        image_url = f"/images/{filename}"
        
        # Use the models to predict which website the trace data belongs to
        # Try both models and use the complex one if available
        predictions = None
        model_used = None
        if complex_model is not None:
            print("Using ComplexModel for prediction")
            predictions = predict_website(trace_data, complex_model)
            model_used = "Complex Model"
        elif simple_model is not None:
            print("Using SimpleModel for prediction")
            predictions = predict_website(trace_data, simple_model)
            model_used = "Simple Model"
        else:
            print("No models available for prediction")
            predictions = {"error": "No models loaded"}
        
        # Add model information to predictions
        if predictions and isinstance(predictions, dict) and "success" in predictions:
            predictions["model_used"] = model_used
            
        # Keep all predicted websites in the output
        if 'predictions' in predictions and isinstance(predictions['predictions'], list):
            # Make sure all predictions are sorted by probability (highest first)
            predictions['predictions'] = sorted(
                predictions['predictions'],
                key=lambda x: x['probability'], 
                reverse=True
            )
        
        # Increment trace_id counter
        trace_ids += 1
        trace_id = trace_ids
        
        # Store in Python arrays (without JSON file references)
        trace_entry = {
            'id': trace_id,
            'trace_data': trace_data,
            'image_url': image_url,
            'timestamp': timestamp,
            'predictions': predictions.get('predictions', []) if 'predictions' in predictions else [],
            'model_used': predictions.get('model_used', 'Unknown')
        }
        
        stored_traces.append(trace_entry)
        stored_heatmaps.append(image_url)  # Store URL instead of base64
        
        # Calculate statistics
        flat_data = trace_array.flatten()
        min_val = int(np.min(flat_data))
        max_val = int(np.max(flat_data))
        
        return jsonify({
            'success': True,
            'trace_id': trace_id,
            'image_url': image_url,
            'timestamp': timestamp,
            'stats': {
                'min': min_val,
                'max': max_val,
                'range': max_val - min_val,
                'samples': trace_array.shape[0],  # Number of time intervals
                'cache_lines': trace_array.shape[1] if len(trace_array.shape) > 1 else 1,  # Number of cache lines
                'total_accesses': int(np.sum(trace_array)),
                'shape': trace_array.shape,
                'heatmap_shape': heatmap_data.shape  # Shape used for heatmap display
            },
            'predictions': predictions.get('predictions', []) if 'predictions' in predictions else [],
            'model_used': predictions.get('model_used', 'Unknown')
        })
        
    except Exception as e:
        print(f"Error in collect_trace: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    """ 
    Implement a clear results endpoint to reset stored data.
    1. Clear stored traces and heatmaps arrays
    2. Return success/error message
    """
    try:
        global stored_traces, stored_heatmaps, trace_ids
        
        # Reset the arrays and counter
        stored_traces = []
        stored_heatmaps = []
        trace_ids = 0
        # Clear images directory
        if os.path.exists('images'):
            shutil.rmtree('images')
        os.makedirs('images', exist_ok=True)
        
        print("Images folder cleared")
        return jsonify({'success': True, 'message': 'All results cleared'})
    
    except Exception as e:
        print(f"Error in clear_results: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_traces', methods=['GET'])
def get_traces():
    """Get all stored traces and heatmaps from the arrays"""
    try:
        global stored_traces
        
        # Return the stored traces sorted by timestamp in descending order
        sorted_traces = sorted(stored_traces, key=lambda x: x['timestamp'])
        
        return jsonify({'traces': sorted_traces})
    
    except Exception as e:
        print(f"Error in get_traces: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/download_traces', methods=['GET'])
def download_traces():
    """Download all trace data as JSON from the arrays"""
    try:
        global stored_traces
        
        # Format the data for download
        all_traces = []
        for trace in sorted(stored_traces, key=lambda x: x['timestamp']):
            all_traces.append({
                'timestamp': trace['timestamp'],
                'data': trace['trace_data'],
                'image_url': trace['image_url'],
            })
        
        return jsonify({'traces': all_traces})
    
    except Exception as e:
        print(f"Error in download_traces: {str(e)}")
        return jsonify({'error': str(e)}), 500

def predict_website(trace_data, model):
    """
    Use the trained model to predict which website the trace data belongs to.
    
    Args:
        trace_data: Raw trace data collected from the frontend
        model: The trained neural network model
    
    Returns:
        Dictionary with predictions and probabilities
    """
    if model is None:
        return {"error": "Model not loaded"}
        
    try:
        # Preprocess the data - ensure it's the right format and length
        trace_array = np.array(trace_data)
        print(f"Original trace array shape: {trace_array.shape}")
        print(f"Original trace array type: {type(trace_array)}")
        
        # Handle dimensionality - if it's 2D, collapse to 1D by summing
        if len(trace_array.shape) == 2:
            trace_array = np.sum(trace_array, axis=1)
            print(f"After collapsing to 1D, shape: {trace_array.shape}")
            
        # Ensure the trace is the right length (1000)
        target_length = 1000  # Must match what the model was trained on
        if len(trace_array) > target_length:
            # Truncate if too long
            trace_array = trace_array[:target_length]
            print(f"Truncated to {target_length}, new shape: {trace_array.shape}")
        elif len(trace_array) < target_length:
            # Pad with zeros if too short
            padding = np.zeros(target_length - len(trace_array))
            trace_array = np.concatenate([trace_array, padding])
            print(f"Padded to {target_length}, new shape: {trace_array.shape}")
        
        # Normalize the data (important for model inference)
        if np.max(trace_array) > 0:  # Avoid division by zero
            trace_array = trace_array / np.max(trace_array)
            print(f"Normalized array, min: {np.min(trace_array)}, max: {np.max(trace_array)}")
        
        # Convert to PyTorch tensor
        input_tensor = torch.tensor(trace_array, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            print(f"Raw model outputs: {outputs}")
            probabilities = F.softmax(outputs, dim=1).squeeze().numpy()
            print(f"Probabilities: {probabilities}")
        
        # Create results
        predictions = []
        for i, (website, prob) in enumerate(zip(WEBSITES, probabilities)):
            predictions.append({
                "website": website,
                "probability": float(prob * 100)  # Convert to percentage
            })
            print(f"Website {i}: {website}, Probability: {float(prob * 100):.2f}%")
        
        # Sort by probability (highest first)
        predictions = sorted(predictions, key=lambda x: x["probability"], reverse=True)
        
        return {
            "success": True,
            "predictions": predictions
        }
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

# Additional endpoints can be implemented here as needed.

if __name__ == '__main__':
    print("Starting Website Fingerprinting Server...")
    print("Open http://localhost:5000 in your browser")
    print("Images will be saved to the 'images' directory")
    app.run(debug=True, host='0.0.0.0')
