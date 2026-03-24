
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import sys

# Add parent directory to sys.path to access engine package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.data_loader import process_pipeline
from engine.model import GuardianHybrid

def evaluate(model, test_loader, device, label_map):
    model.eval()
    all_preds = []
    all_targets = []
    
    print("\nStarting Inference on 2018 Data...")
    print("\nStarting Inference on 2018 Data...")
    
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            if i % 500 == 0:
                 print(f"Batch {i}/{len(test_loader)}...")
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass: classify
            probs = model(data, mode='classify')
            _, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_targets, all_preds)
    print(f"\nOverall Accuracy: {acc*100:.2f}%")
    
    idx_to_label = {v: k for k, v in label_map.items()}
    present_indices = sorted(list(set(all_targets) | set(all_preds)))
    target_names = [idx_to_label.get(i, f"Class {i}") for i in present_indices]
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, labels=present_indices, target_names=target_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds, labels=present_indices)
    print(cm)
    
    # Binary
    print("\n=== Binary Classification Performance (Benign vs Malicious) ===")
    benign_idx = label_map['BENIGN']
    binary_targets = [0 if t == benign_idx else 1 for t in all_targets]
    binary_preds = [0 if p == benign_idx else 1 for p in all_preds]
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(binary_targets, binary_preds, average=None, labels=[0, 1])
    
    print(f"Benign      - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}, Count: {support[0]}")
    print(f"Attack      - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}, Count: {support[1]}")
    
    tn, fp, fn, tp = confusion_matrix(binary_targets, binary_preds).ravel()
    print(f"TN: {tn}, FP: {fp}, TP: {tp}, FN: {fn}")
    print(f"Binary Acc: {(tp+tn)/(tp+tn+fp+fn)*100:.2f}%")

    # Metrics requested by user
    benign_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    attack_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"Found Benign / Benign Rate: {benign_rate*100:.2f}%")
    print(f"Found Not Benign / Not Benign Rate: {attack_rate*100:.2f}%")

RENAME_MAP_2018 = {
    'Dst Port': ' Destination Port',
    'Tot Fwd Pkts': ' Total Fwd Packets',
    'Tot Bwd Pkts': ' Total Backward Packets',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
    'TotLen Bwd Pkts': ' Total Length of Bwd Packets',
    'Fwd Pkt Len Max': ' Fwd Packet Length Max',
    'Fwd Pkt Len Min': ' Fwd Packet Length Min',
    'Fwd Pkt Len Mean': ' Fwd Packet Length Mean',
    'Fwd Pkt Len Std': ' Fwd Packet Length Std',
    'Bwd Pkt Len Max': 'Bwd Packet Length Max',
    'Bwd Pkt Len Min': ' Bwd Packet Length Min',
    'Bwd Pkt Len Mean': ' Bwd Packet Length Mean',
    'Bwd Pkt Len Std': ' Bwd Packet Length Std',
    'Flow Byts/s': 'Flow Bytes/s',
    'Flow Pkts/s': ' Flow Packets/s',
    'Flow IAT Mean': ' Flow IAT Mean',
    'Flow IAT Std': ' Flow IAT Std',
    'Flow IAT Max': ' Flow IAT Max',
    'Flow IAT Min': ' Flow IAT Min',
    'Fwd IAT Tot': 'Fwd IAT Total',
    'Fwd IAT Mean': ' Fwd IAT Mean',
    'Fwd IAT Std': ' Fwd IAT Std',
    'Fwd IAT Max': ' Fwd IAT Max',
    'Fwd IAT Min': ' Fwd IAT Min',
    'Bwd IAT Tot': 'Bwd IAT Total',
    'Bwd IAT Mean': ' Bwd IAT Mean',
    'Bwd IAT Std': ' Bwd IAT Std',
    'Bwd IAT Max': ' Bwd IAT Max',
    'Bwd IAT Min': ' Bwd IAT Min',
    'Fwd PSH Flags': 'Fwd PSH Flags',
    'Bwd PSH Flags': ' Bwd PSH Flags',
    'Fwd URG Flags': ' Fwd URG Flags',
    'Bwd URG Flags': ' Bwd URG Flags',
    'Fwd Header Len': ' Fwd Header Length',
    'Bwd Header Len': ' Bwd Header Length',
    'Fwd Pkts/s': 'Fwd Packets/s',
    'Bwd Pkts/s': ' Bwd Packets/s',
    'Pkt Len Min': ' Min Packet Length',
    'Pkt Len Max': ' Max Packet Length',
    'Pkt Len Mean': ' Packet Length Mean',
    'Pkt Len Std': ' Packet Length Std',
    'Pkt Len Var': ' Packet Length Variance',
    'FIN Flag Cnt': 'FIN Flag Count',
    'SYN Flag Cnt': ' SYN Flag Count',
    'RST Flag Cnt': ' RST Flag Count',
    'PSH Flag Cnt': ' PSH Flag Count',
    'ACK Flag Cnt': ' ACK Flag Count',
    'URG Flag Cnt': ' URG Flag Count',
    'CWE Flag Count': ' CWE Flag Count',
    'ECE Flag Cnt': ' ECE Flag Count',
    'Down/Up Ratio': ' Down/Up Ratio',
    'Pkt Size Avg': ' Average Packet Size',
    'Fwd Seg Size Avg': ' Avg Fwd Segment Size',
    'Bwd Seg Size Avg': ' Avg Bwd Segment Size',
    'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',
    'Fwd Pkts/b Avg': ' Fwd Avg Packets/Bulk',
    'Fwd Blk Rate Avg': ' Fwd Avg Bulk Rate',
    'Bwd Byts/b Avg': ' Bwd Avg Bytes/Bulk',
    'Bwd Pkts/b Avg': ' Bwd Avg Packets/Bulk',
    'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
    'Subflow Fwd Pkts': 'Subflow Fwd Packets',
    'Subflow Fwd Byts': ' Subflow Fwd Bytes',
    'Subflow Bwd Pkts': ' Subflow Bwd Packets',
    'Subflow Bwd Byts': ' Subflow Bwd Bytes',
    'Init Fwd Win Byts': 'Init_Win_bytes_forward',
    'Init Bwd Win Byts': ' Init_Win_bytes_backward',
    'Fwd Act Data Pkts': ' act_data_pkt_fwd',
    'Fwd Seg Size Min': ' min_seg_size_forward',
    'Active Mean': 'Active Mean',
    'Active Std': ' Active Std',
    'Active Max': ' Active Max',
    'Active Min': ' Active Min',
    'Idle Mean': 'Idle Mean',
    'Idle Std': ' Idle Std',
    'Idle Max': ' Idle Max',
    'Idle Min': ' Idle Min',
    'Label': ' Label',
    'Dst IP': ' Destination IP',
    'Src IP': ' Source IP',
    'Src Port': ' Source Port',
    'Flow ID': 'Flow ID',
    'Protocol': ' Protocol',
    'Timestamp': ' Timestamp'
}

def process_chunk(chunk, loader, fill_stats, scaler_path):
    """Process a single DataFrame chunk: rename, clean, scale, sequence. Returns (sequences, labels) or None."""
    chunk = chunk.rename(columns=RENAME_MAP_2018)

    if ' Fwd Header Length' in chunk.columns and ' Fwd Header Length.1' not in chunk.columns:
        chunk[' Fwd Header Length.1'] = chunk[' Fwd Header Length']

    cleaned_df, labels = loader.clean_data(chunk, mode='train_classifier', fill_stats=fill_stats)
    del chunk

    if cleaned_df.empty or labels is None or len(labels) == 0:
        return None, None

    loader.load_scaler(scaler_path)
    data_scaled = loader.transform(cleaned_df)
    del cleaned_df

    sequences, seq_labels = loader.create_sequences(data_scaled, labels)
    del data_scaled, labels

    if sequences.size == 0:
        return None, None

    return sequences, seq_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--chunk_size', type=int, default=30000, help="Rows per chunk to keep memory low")
    args = parser.parse_args()

    import gc
    import glob
    import pandas as pd
    import joblib
    from torch.utils.data import DataLoader, TensorDataset
    from engine.data_loader import GuardianDataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ids2018_path = os.path.join(base_dir, 'data', 'test', 'ids2018')
    scaler_path = os.path.join(base_dir, 'checkpoints', 'guardian_scaler.pkl')
    model_path = os.path.join(base_dir, 'checkpoints', 'guardian_complete.pth')

    if not os.path.exists(model_path):
        print("Model not found.")
        return

    csv_files = sorted(glob.glob(os.path.join(ids2018_path, "*.csv")))
    if not csv_files:
        print("No CSV files found.")
        return

    # Load scaler fill_stats once
    fill_stats = None
    if os.path.exists(scaler_path):
        s = joblib.load(scaler_path)
        if hasattr(s, 'feature_names_in_') and hasattr(s, 'data_max_'):
            fill_stats = dict(zip(s.feature_names_in_, s.data_max_))
        del s

    label_map = {
        'BENIGN': 0, 'DDoS': 1, 'PortScan': 2, 'Web Attack': 3, 'Bot': 4
    }

    model = None
    all_preds = []
    all_targets = []
    total_processed = 0

    for file_idx, csv_file in enumerate(csv_files):
        fname = os.path.basename(csv_file)
        print(f"\n{'='*60}")
        print(f"Processing file {file_idx+1}/{len(csv_files)}: {fname}")
        print(f"{'='*60}")

        chunk_num = 0
        file_samples = 0

        for chunk in pd.read_csv(csv_file, encoding='cp1252', low_memory=False, chunksize=args.chunk_size):
            chunk_num += 1
            loader = GuardianDataLoader(seq_len=10)

            sequences, seq_labels = process_chunk(chunk, loader, fill_stats, scaler_path)
            del chunk, loader
            gc.collect()

            if sequences is None:
                print(f"  Chunk {chunk_num}: no valid data, skipping.")
                continue

            n_features = sequences.shape[2]

            # Init model on first successful chunk
            if model is None:
                print(f"Loading Model from {model_path}...")
                model = GuardianHybrid(input_dim=n_features).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()

            # Run inference on this chunk
            seq_copy = np.array(sequences, copy=True)
            lbl_copy = np.array(seq_labels, copy=True)
            del sequences, seq_labels
            gc.collect()

            dataset = TensorDataset(torch.FloatTensor(seq_copy), torch.FloatTensor(lbl_copy))
            del seq_copy, lbl_copy
            test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            with torch.no_grad():
                for data, lbls in test_loader:
                    data = data.to(device)
                    probs = model(data, mode='classify')
                    _, preds = torch.max(probs, 1)
                    all_preds.extend(preds.cpu().tolist())
                    all_targets.extend(lbls.tolist())

            chunk_samples = len(dataset)
            file_samples += chunk_samples
            del dataset, test_loader
            torch.cuda.empty_cache()
            gc.collect()

            print(f"  Chunk {chunk_num}: {chunk_samples} sequences processed.")

        total_processed += file_samples
        print(f"File complete: {file_samples} sequences from {fname}. Running total: {total_processed}")

    # Final combined metrics
    if not all_preds:
        print("No data was processed.")
        return

    print(f"\n{'='*60}")
    print(f"COMBINED RESULTS ACROSS ALL FILES ({len(all_targets)} total samples)")
    print(f"{'='*60}")

    acc = accuracy_score(all_targets, all_preds)
    print(f"\nOverall Accuracy: {acc*100:.2f}%")

    idx_to_label = {v: k for k, v in label_map.items()}
    present_indices = sorted(list(set(all_targets) | set(all_preds)))
    target_names = [idx_to_label.get(int(i), f"Class {int(i)}") for i in present_indices]

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, labels=present_indices, target_names=target_names, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds, labels=present_indices)
    print(cm)

    print("\n=== Binary Classification Performance (Benign vs Malicious) ===")
    benign_idx = label_map['BENIGN']
    binary_targets = [0 if t == benign_idx else 1 for t in all_targets]
    binary_preds = [0 if p == benign_idx else 1 for p in all_preds]

    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(binary_targets, binary_preds, average=None, labels=[0, 1])

    print(f"Benign      - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}, Count: {support[0]}")
    print(f"Attack      - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}, Count: {support[1]}")

    tn, fp, fn, tp = confusion_matrix(binary_targets, binary_preds).ravel()
    print(f"TN: {tn}, FP: {fp}, TP: {tp}, FN: {fn}")
    print(f"Binary Acc: {(tp+tn)/(tp+tn+fp+fn)*100:.2f}%")

    benign_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    attack_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"Found Benign / Benign Rate: {benign_rate*100:.2f}%")
    print(f"Found Not Benign / Not Benign Rate: {attack_rate*100:.2f}%")

if __name__ == "__main__":
    main()
