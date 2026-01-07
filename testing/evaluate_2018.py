
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ids2018_path = os.path.join(base_dir, 'data', 'test', 'ids2018')
    scaler_path = os.path.join(base_dir, 'checkpoints', 'guardian_scaler.pkl')
    model_path = os.path.join(base_dir, 'checkpoints', 'guardian_complete.pth')
    
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    # Load 2018 Data with 'ids2018' type to trigger column mapping
    print("Loading IDS-2018 Test Data (Limit: 100k rows/file)...")
    seqs, labels, loader = process_pipeline(ids2018_path, scaler_path=scaler_path, mode='train_classifier', dataset_type='ids2018', limit=10000)
    
    if seqs.size == 0:
        print("No data found.")
        return
        
    n_features = seqs.shape[2]
    
    print(f"Loading Model from {model_path}...")
    model = GuardianHybrid(input_dim=n_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.Tensor(seqs), torch.Tensor(labels))
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2017 Label Map
    label_map = {
        'BENIGN': 0, 'DDoS': 1, 'PortScan': 2, 'Web Attack': 3, 'Bot': 4
    }
    
    evaluate(model, test_loader, device, label_map)

if __name__ == "__main__":
    main()
