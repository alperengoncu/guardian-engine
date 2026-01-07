# Guardian Engine

**Guardian NIDS** is a Hybrid, Identity-Free Network Intrusion Detection System powered by an LSTM-based Autoencoder and Classifier architecture. It is designed to detect known and unknown network attacks by analyzing flow-based features, intentionally ignoring identity fields (IPs, Ports) to ensure generalization across different networks.

## Features
- **Hybrid Architecture**: Combines an Autoencoder (for anomaly detection on Benign traffic) and a Classifier (for multi-class attack identification).
- **Identity-Free**: explicitly drops source/destination IPs and ports to prevent overfitting to specific network environments.
- **Multi-Dataset Support**: Built for CIC-IDS-2017 but includes compatibility layers for CIC-IDS-2018.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/alperengoncu/guardian-engine.git
    cd guardian-engine
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install PyTorch (CUDA Support)**:
    > **Note**: Default pip installs might not include CUDA support. To enable GPU acceleration, please install the version matching your CUDA driver from the [official PyTorch website](https://pytorch.org/get-started/locally/).
    
    Example (adjust for your CUDA version):
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## Project Structure & Scripts

### `engine/` - Core Logic
This directory contains the training and modeling backend.

- **`train.py`**: The main training pipeline. It runs in two phases:
    - **Phase 1 (Autoencoder)**: Trains on BENIGN traffic only to learn normal patterns.
    - **Phase 2 (Classifier)**: Freezes the encoder and trains the classifier head on all traffic types.
    - *Usage*: `python engine/train.py --phase all`
- **`model.py`**: Defines the `GuardianHybrid` PyTorch model (Encoder, Decoder, Classifier).
- **`data_loader.py`**: Handles all data preprocessing.
    - Ingests CSVs from `data/`.
    - Drops identity columns (IPs, Ports, Timestamps).
    - Normalizes data (MinMaxScaling).
    - Generates sliding window sequences for LSTM input.
    - Maps CIC-IDS-2018 columns to the 2017 schema automatically.

### `testing/` - Evaluation Scripts
Scripts to validate the model's performance.

- **`evaluate.py`**:
    - Evaluates the model on the **CIC-IDS-2017** test set.
    - outputs detailed classification reports and confusion matrices.
- **`test_samples.py`**:
    - Randomly selects 50 samples from the **2017** dataset.
    - Runs inference and displays a "True vs Predicted" comparison table.
- **`evaluate_2018.py`**:
    - Evaluates the model on the **CIC-IDS-2018** dataset.
    - Includes memory optimizations (row limits) to handle the massive dataset size.
- **`test_samples_2018.py`**:
    - Randomly selects 50 samples from the **2018** dataset for quick verification.

### `data/` and `checkpoints/`
- **`data/`**: Store your datasets here (e.g., `data/train/ids-2017`, `data/test/ids2018`).
- **`checkpoints/`**: Stores trained model weights (`.pth`) and scalers (`.pkl`).

## Usage Example

**Training:**
```bash
cd engine
python train.py --phase all
```

**Testing (2017):**
```bash
cd testing
python evaluate.py
```

**Testing (2018):**
```bash
cd testing
python evaluate_2018.py
```
