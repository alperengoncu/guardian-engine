
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
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass: classify
            probs = model(data, mode='classify')
            _, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_targets, all_preds)
    print(f"\nOverall Accuracy: {acc*100:.2f}%")
    
    # Invert label map for nice printing
    # label_map is {'BENIGN': 0, ...} -> {0: 'BENIGN', ...}
    idx_to_label = {v: k for k, v in label_map.items()}
    # Filter based on present labels in test set to avoid error if some classes missing
    present_indices = sorted(list(set(all_targets) | set(all_preds)))
    target_names = [idx_to_label.get(i, f"Class {i}") for i in present_indices]
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, labels=present_indices, target_names=target_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds, labels=present_indices)
    print(cm)
    
    # --- Binary Metrics (Benign vs Malicious) ---
    print("\n=== Binary Classification Performance (Benign vs Malicious) ===")
    
    # Identify Benign Index (from label_map)
    benign_idx = label_map['BENIGN']
    
    # Convert properly to binary arrays
    # 0 for Benign, 1 for Malicious
    binary_targets = [0 if t == benign_idx else 1 for t in all_targets]
    binary_preds = [0 if p == benign_idx else 1 for p in all_preds]
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(binary_targets, binary_preds, average=None, labels=[0, 1])
    
    print(f"Benign (Normal)     - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}, Count: {support[0]}")
    print(f"Non-Benign (Attack) - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}, Count: {support[1]}")
    
    # Detailed binary counts
    tn, fp, fn, tp = confusion_matrix(binary_targets, binary_preds).ravel()
    print(f"\nBinary Confusion Matrix:")
    print(f"True Benign (TN): {tn}")
    print(f"False Benign (Missed Attack - FN): {fn}")
    print(f"True Malicious (TP): {tp}")
    print(f"False Malicious (False Alarm - FP): {fp}")
    
    binary_acc = (tp + tn) / (tp + tn + fp + fn)
    print(f"\nBinary Accuracy: {binary_acc*100:.2f}%")
    
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Base dir: testing/evaluate.py -> .. -> root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    primary_path = os.path.join(base_dir, 'data', 'train', 'ids-2017')
    scaler_path = os.path.join(base_dir, 'checkpoints', 'guardian_scaler.pkl')
    model_path = os.path.join(base_dir, 'checkpoints', 'guardian_complete.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train first.")
        return

    # Load Data
    # Utilizing 'train_classifier' mode to load everything with labels
    print("Loading Test Data...")
    seqs, labels, loader = process_pipeline(primary_path, scaler_path=scaler_path, mode='train_classifier')
    
    if seqs.size == 0:
        print("No data found. Exiting.")
        return
        
    n_features = seqs.shape[2]
    
    # Load Model
    print(f"Loading Model from {model_path}...")
    model = GuardianHybrid(input_dim=n_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Create DataLoader
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.Tensor(seqs), torch.Tensor(labels))
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Label Map (manual copy from data_loader or imported if refactored)
    # Ideally should be consistent
    label_map = {
        'BENIGN': 0,
        'DDoS': 1,
        'PortScan': 2,
        'Web Attack': 3,
        'Bot': 4
    }
    
    evaluate(model, test_loader, device, label_map)

if __name__ == "__main__":
    main()
