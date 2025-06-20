# Website Fingerprinting via Side-Channel Attacks

A comprehensive side-channel attack implementation that identifies websites users visit in other tabs using cache timing measurements. Achieves 98.5% accuracy on individual datasets and 81.7% on collaborative datasets.

## Project Resources

- **Notebook**: [Web Fingerprinting Analysis](https://www.kaggle.com/code/ahmmadnurswapnil/web-fringerprint)
- **Dataset**: [Original Training Data](https://www.kaggle.com/datasets/ahmmadnurswapnil/train-data)
- **Merged Dataset**: [Combined Training Data](https://www.kaggle.com/datasets/ahmmadnurswapnil/train-data-all)
- **Contribution Files**: [Google Drive](https://drive.google.com/drive/folders/15UtiDYM8AQor5M_YyuxfYY4O0t5ELXwY)

## Core Components

1. **Timing Analysis**: JavaScript-based high-resolution timing measurements
2. **Sweep Counting**: Cache eviction technique for side-channel access
3. **Automated Collection**: Selenium WebDriver for collecting website fingerprints
4. **ML Classification**: CNN models that identify websites from timing patterns
5. **Real-time Detection**: Live website identification in the browser

## Quick Start

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   # OR manually install:
   pip install sqlalchemy selenium webdriver-manager flask numpy matplotlib
   ```

2. **Test your setup**:

   ```bash
   python test_setup.py  # Optional validation
   ```

3. **Task 1 - Timing Warmup**:

   ```bash
   python app.py
   # Open http://localhost:5000
   # Click "Collect Latency Data"
   ```

4. **Task 2 - Sweep Counting Attack**:

   ```bash
   python app.py
   # Open http://localhost:5000
   # Open target website in another tab
   # Click "Collect trace Data"
   # Observe heatmap patterns
   ```

5. **Task 3 - Automated Data Collection**:

   ```bash
   # Terminal 1: Start Flask server
   python app.py

   # Terminal 2: Run automated collection
   python collect.py
   ```

## Implementation Status

### ✅ All Tasks Completed

- **Task 1 (10%)**: Browser timing precision measurement
  - `readNlines(n)` function implemented in `static/warmup.js`
  - Uses `performance.now()` for high-resolution timing
  - Tests with different cache line counts (1 to 10,000,000)

- **Task 2 (35%)**: Sweep counting attack with visualization
  - `sweep(P)` function implemented in `static/worker.js`
  - Allocates LLC-sized buffer (8MB default)
  - Performs sweep counting for 10 seconds with 10ms windows
  - Generates heatmaps showing cache access patterns

- **Task 3 (20%)**: Automated data collection
  - Complete Selenium automation with browser control
  - SQLite database for persistent storage
  - Error handling and retry mechanisms
  - Progress tracking and clean shutdown

- **Task 4 (35%)**: Machine learning classification
  - PyTorch neural network implementation
  - Both simple and complex model architectures
  - Data preprocessing and normalization
  - Model evaluation and result visualization
  - **Results**: 95.5% accuracy (Simple Model), 98.5% accuracy (Complex Model)

- **Bonus Task 2 (20%)**: Collaborative dataset collection
  - Large-scale data collection with 100,000+ traces
  - Extended to 5 websites including Chaldal and Dhaka Tribune
  - **Results**: 74.6% accuracy (Simple Model), 81.7% accuracy (Complex Model)

- **Bonus Task 3 (15%)**: Real-time website detection
  - Live detection of websites from adjacent tabs
  - Visual probability display for website identification
  - Dynamic UI showing confidence levels
  - Support for multiple trained models
  - **Demonstration**: 86.5% confidence detection in real-time

## Features

- **Real-time Website Detection**: Identifies websites being visited in adjacent tabs
- **Visual Heatmaps**: Displays cache access patterns as color-coded heatmaps
- **Probability Analysis**: Shows confidence levels for website prediction
- **Multiple Models**: Implements both simple and complex neural network models
- **Dynamic UI**: Shows only predictions with meaningful confidence levels
- **Filtered Results**: Only displays websites with non-zero confidence

## Supported Websites

The system can identify the following websites:
- https://cse.buet.ac.bd/moodle/
- https://google.com
- https://prothomalo.com
- https://www.chaldal.com
- https://www.dhakatribune.com

## Machine Learning Models

Two main model architectures were implemented:

1. **SimpleModel**: Basic neural network with 2 convolutional layers and fully connected layers
2. **ComplexModel**: Deeper architecture with 3 convolutional layers, batch normalization, and dropout

Both models were trained on collected traces from the 5 target websites, achieving high accuracy in website identification:

### Performance Results

| Model Type | Individual Dataset | Collaborative Dataset |
|------------|-------------------|----------------------|
| Simple Model | 95.5% accuracy | 74.6% accuracy |
| Complex Model | 98.5% accuracy | 81.7% accuracy |

The Complex Model consistently outperforms the Simple Model, with the collaborative dataset demonstrating scalability challenges in real-world scenarios.

## Automated Data Collection (Task 3)

The `collect.py` script implements automated website fingerprinting data collection:

- **Selenium automation**: Automatically opens websites and collects traces
- **Database storage**: Uses SQLite to reliably store collected traces
- **Error handling**: Robust error handling for long-running collection
- **Progress tracking**: Shows collection progress and handles interruptions
- **Clean shutdown**: Saves data before exiting on interruption

### Configuration

Edit `collect.py` to customize:

- `WEBSITES`: List of websites to fingerprint (default: BUET Moodle, Google, Prothomalo)
- `TRACES_PER_SITE`: Number of traces to collect per website (default: 10 for testing, spec allows 1000+)
- `FINGERPRINTING_URL`: Flask server URL (default: http://localhost:5000)

### Database Structure

The SQLite database (`webfingerprint.db`) contains:

- `fingerprints`: Stores trace data with website info and timestamps
- `collection_stats`: Tracks collection progress per website

### Usage Tips

1. **Start small**: Test with 10 traces per site first (current default)
2. **Monitor progress**: Check console output for collection status
3. **Handle interruptions**: Use Ctrl+C to gracefully stop collection
4. **Export data**: Final dataset is saved to `dataset.json`
5. **Resume collection**: Restart script to continue from where it left off
6. **Browser requirements**: Chrome browser and ChromeDriver (auto-installed)

7. **Run the Flask server**:

   ```bash
   python app.py
   ```

8. **Open the web interface**:

   - Navigate to http://127.0.0.1:5000
   - Click buttons to collect timing data and traces

9. **Automated data collection**:

   ```bash
   python collect.py
   ```

10. **Train ML models**:
    ```bash
    python train.py
    ```

## Files Structure

```
template/
├── app.py                 # Flask server with endpoints
├── collect.py             # Selenium automation script
├── database.py            # SQLite database management
├── train.py               # Machine learning model training
├── merged_train.py        # Training on merged datasets
├── static/
│   ├── index.html         # Main UI
│   ├── index.js           # Frontend JavaScript
│   ├── warmup.js          # Task 1 timing worker
│   └── worker.js          # Task 2 sweep counting worker
├── images/                # Generated heatmaps
├── results/               # Trained models and metrics
│   └── saved_models/      # Trained SimpleModel and ComplexModel
└── dataset.json           # Exported trace data
```

## Performance

- **Individual Dataset**: Simple Model (95.5%), Complex Model (98.5%)
- **Collaborative Dataset**: Simple Model (74.6%), Complex Model (81.7%)  
- **Websites**: Successfully tested with 5 different websites
- **Data Collection**: 12-hour automated collection process, up to 1000 traces per website
- **Real-time Detection**: Live classification with 86.5% confidence demonstration

## Security Implications

This demonstrates how **co-located attacks** can compromise user privacy by:

- Identifying visited websites without network access
- Working across browser tabs and processes
- Bypassing traditional security measures

The implementation shows how side-channel attacks can be used to detect user browsing activity without direct access to sensitive information, highlighting the importance of side-channel protections in browser security.

**Note**: This project was developed for educational purposes in computer security research.
