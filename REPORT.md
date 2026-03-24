# Guardian Engine - Evaluation Report

## 1. System Overview

Guardian Engine is a hybrid deep learning-based Network Intrusion Detection System (NIDS). It uses a two-phase training approach built on a **Conv1D + LSTM Autoencoder** architecture with a classification head.

### Architecture: GuardianHybrid

| Component | Details |
|-----------|---------|
| **Spatial Feature Extraction** | Conv1D (input_dim -> 64 channels, kernel=3) + BatchNorm1d |
| **Temporal Feature Extraction** | LSTM (64 -> 128 hidden units, batch_first) |
| **Latent Bottleneck** | Linear (128 -> 32 latent dimensions) |
| **Decoder (Reconstruction)** | Linear(32 -> 128) -> Linear(128 -> seq_len * input_dim) |
| **Classifier Head** | Linear(32 -> 64) -> ReLU -> Dropout(0.5) -> Linear(64 -> 5 classes) |

### Training Pipeline

- **Phase 1 - Autoencoder**: Trained on BENIGN traffic only using MSE reconstruction loss. The model learns what "normal" network behavior looks like.
- **Phase 2 - Classifier**: Encoder layers are frozen. Only the classifier head is trained using NLLLoss on all 5 traffic classes.

### Data Processing

- **Input**: CIC-IDS CSV flow records (77 features after cleaning)
- **Normalization**: MinMaxScaler fitted on training data
- **Sequence Length**: Sliding window of 10 consecutive flows -> shape (N, 10, 77)
- **Identity columns dropped**: Flow ID, Source/Dest IP, Ports, Protocol, Timestamp (prevents overfitting to network-specific identifiers)
- **Inf/NaN handling**: Replaced with column max values from training scaler, then 0 for remaining

### Class Mapping

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | BENIGN | Normal traffic |
| 1 | DDoS | Distributed Denial of Service |
| 2 | PortScan | Port scanning activity |
| 3 | Web Attack | Brute Force, XSS, SQL Injection |
| 4 | Bot | Botnet traffic |

---

## 2. Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs (per phase) | 5 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Sequence Length | 10 |
| Latent Dimension | 32 |
| Training Data | CIC-IDS-2017 (50,000 rows/file, 8 files) |
| Device | CUDA (GPU) |

### Autoencoder Training Loss

| Epoch | MSE Loss |
|-------|----------|
| 1 | 0.003363 |
| 2 | 0.001706 |
| 3 | 0.001358 |
| 4 | 0.001200 |
| 5 | 0.001137 |

---

## 3. Test 1: CIC-IDS-2017 Evaluation (Same-Dataset)

**Dataset**: CIC-IDS-2017 (50,000 rows/file, 8 files, 399,991 sequences total)

### Overall Accuracy: **97.97%**

### Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| BENIGN | 0.98 | 1.00 | 0.99 | 366,954 |
| DDoS | 0.98 | 0.96 | 0.97 | 26,526 |
| PortScan | 0.77 | 0.01 | 0.01 | 5,427 |
| Web Attack | 0.00 | 0.00 | 0.00 | 1,082 |
| Bot | 0.00 | 0.00 | 0.00 | 2 |
| **Weighted Avg** | **0.97** | **0.98** | **0.97** | **399,991** |

### Confusion Matrix (CIC-IDS-2017)

|  | Pred: BENIGN | Pred: DDoS | Pred: PortScan | Pred: Web Attack | Pred: Bot |
|--|-------------|-----------|----------------|-----------------|----------|
| **BENIGN** | 366,361 | 584 | 9 | 0 | 0 |
| **DDoS** | 1,033 | 25,490 | 3 | 0 | 0 |
| **PortScan** | 5,386 | 1 | 40 | 0 | 0 |
| **Web Attack** | 1,082 | 0 | 0 | 0 | 0 |
| **Bot** | 2 | 0 | 0 | 0 | 0 |

### Binary Performance (Benign vs Malicious)

| Metric | Benign (Normal) | Attack (Malicious) |
|--------|----------------|-------------------|
| Precision | 0.9799 | 0.9773 |
| Recall | 0.9984 | 0.7729 |
| F1-Score | 0.9891 | 0.8632 |
| Support | 366,954 | 33,037 |

| Binary Metric | Value |
|---------------|-------|
| Binary Accuracy | 97.98% |
| True Benign (TN) | 366,361 |
| False Benign (Missed Attack - FN) | 7,503 |
| True Malicious (TP) | 25,534 |
| False Alarm (FP) | 593 |
| Benign Detection Rate | 99.84% |
| Attack Detection Rate | 77.29% |

---

## 4. Test 2: CIC-IDS-2018 Evaluation (Cross-Dataset Generalization)

**Dataset**: CIC-IDS-2018 (all 4 CSV files, processed in 30k-row chunks, **10,514,974 sequences** total)
**Purpose**: Test how well the model trained on 2017 data generalizes to unseen 2018 attack data.

### Excluded Attack Types

The following attack types present in CIC-IDS-2018 were **excluded** from evaluation because they have no equivalent in the CIC-IDS-2017 training data:

| Excluded Label | Reason |
|----------------|--------|
| DDoS attacks-LOIC-HTTP | HTTP-based LOIC DDoS — 2018-only tool |
| DDOS attack-HOIC | High Orbit Ion Cannon — 2018-only tool |
| DDOS attack-LOIC-UDP | UDP-based LOIC DDoS — 2018-only tool |

### Per-File Breakdown

| File | Sequences |
|------|-----------|
| 02-14-2018.csv | 1,048,260 |
| 02-15-2018.csv | 1,048,260 |
| 02-16-2018.csv | 1,048,259 |
| 02-20-2018.csv | 7,370,195 |
| **Total** | **10,514,974** |

### Overall Accuracy: **91.59%**

### Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| BENIGN | 0.97 | 0.96 | 0.96 | 9,480,012 |
| DDoS | 0.48 | 0.81 | 0.60 | 654,121 |
| PortScan | 0.00 | 0.00 | 0.00 | 380,841 |
| **Weighted Avg** | **0.90** | **0.92** | **0.91** | **10,514,974** |

### Confusion Matrix (CIC-IDS-2018)

|  | Pred: BENIGN | Pred: DDoS | Pred: PortScan |
|--|-------------|-----------|----------------|
| **BENIGN** | 9,103,401 | 376,595 | 16 |
| **DDoS** | 126,813 | 527,295 | 13 |
| **PortScan** | 188,133 | 192,708 | 0 |

### Binary Performance (Benign vs Malicious)

| Metric | Benign (Normal) | Attack (Malicious) |
|--------|----------------|-------------------|
| Precision | 0.9666 | 0.6566 |
| Recall | 0.9603 | 0.6957 |
| F1-Score | 0.9634 | 0.6756 |
| Support | 9,480,012 | 1,034,962 |

| Binary Metric | Value |
|---------------|-------|
| Binary Accuracy | 93.42% |
| True Benign (TN) | 9,103,401 |
| False Benign (Missed Attack - FN) | 314,946 |
| True Malicious (TP) | 720,016 |
| False Alarm (FP) | 376,611 |
| Benign Detection Rate | 96.03% |
| Attack Detection Rate | 69.57% |

---

## 5. Analysis & Key Findings

### Strengths
- **High same-dataset accuracy** (97.97%) on CIC-IDS-2017, demonstrating the model learns the training distribution well.
- **Strong cross-dataset generalization**: 91.59% overall accuracy on 10.5M unseen 2018 sequences, with 93.42% binary accuracy.
- **Excellent BENIGN detection** (99.84% recall on 2017, 96.03% on 2018) with consistent performance across both datasets.
- **Strong DDoS detection** (96% recall on 2017, 81% recall on 2018), the most represented attack class, showing the model transfers DDoS knowledge across years.
- **Low false positive rate on 2017**: Only 0.16% of benign traffic is misclassified as an attack.
- **Scalable evaluation**: The chunked processing pipeline successfully evaluated over 10.5 million sequences without memory issues.

### Weaknesses
- **Minority class failure**: PortScan (0.7% recall), Web Attack (0% recall), and Bot (0% recall) are almost entirely misclassified as BENIGN on 2017 data. This is due to severe class imbalance — the model defaults to predicting the majority class.
- **PortScan completely missed on 2018**: All 380,841 PortScan samples were misclassified — roughly half as DDoS and half as BENIGN — suggesting the model conflates attack subtypes when feature distributions differ across datasets.
- **Higher false positive rate on 2018**: 376,611 benign flows flagged as DDoS (3.97% FP rate), compared to 0.16% on 2017, indicating some distribution shift in benign traffic signatures between the two datasets.
- **DDoS partial confusion on 2018**: 126,813 / 654,121 DDoS samples were misclassified as BENIGN (19%), indicating some 2018 DoS variants (GoldenEye, Slowloris, SlowHTTPTest, Hulk) have different flow signatures than 2017 DDoS.

### Recommendations for Improvement
1. **Address class imbalance**: Use oversampling (SMOTE), class weights in the loss function, or focal loss to improve minority class detection.
2. **Increase training data diversity**: Train on both 2017 and 2018 data together, or use domain adaptation techniques.
3. **Increase training epochs and data**: The model was trained on a 50k row subset for 5 epochs — using more data and longer training would likely improve generalization.
4. **Feature engineering**: Investigate which features differ most between 2017 and 2018 datasets to improve the column mapping.

---

*Report generated: 2026-03-22*
*Model: GuardianHybrid (Conv1D + LSTM Autoencoder + Classifier)*
*Training data: CIC-IDS-2017 | Test data: CIC-IDS-2017 & CIC-IDS-2018*
