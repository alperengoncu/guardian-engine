
import os
import random
import numpy as np
import torch
import pandas as pd
import sys

# Add parent directory to sys.path to access engine package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.data_loader import process_pipeline
from engine.model import GuardianHybrid

def test_random_samples_2018():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ids2018_path = os.path.join(base_dir, 'data', 'test', 'ids2018')
    scaler_path = os.path.join(base_dir, 'checkpoints', 'guardian_scaler.pkl')
    model_path = os.path.join(base_dir, 'checkpoints', 'guardian_complete.pth')
    
    if not os.path.exists(model_path):
        print("Model not found.")
        return

    print("Loading IDS-2018 Data for Sampling (First file only)...")
    seqs, labels, loader = process_pipeline(ids2018_path, scaler_path=scaler_path, mode='train_classifier', dataset_type='ids2018', max_files=1)
    
    if seqs.size == 0:
        print("No data loaded.")
        return
        
    n_features = seqs.shape[2]
    total_samples = len(seqs)
    print(f"Total samples available: {total_samples}")
    
    model = GuardianHybrid(input_dim=n_features).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    idx_to_label = {0: 'BENIGN', 1: 'DDoS', 2: 'PortScan', 3: 'Web Attack', 4: 'Bot'}
    
    indices = random.sample(range(total_samples), 50)
    
    print("\n=== Random Sample Test 2018 (50 Packets) ===")
    print(f"{'ID':<6} | {'True Label':<12} | {'Predicted':<12} | {'Conf%':<6} | {'Status':<10}")
    print("-" * 60)
    
    correct_count = 0
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = seqs[idx]
            true_label_idx = int(labels[idx])
            true_label_str = idx_to_label.get(true_label_idx, f"Unk({true_label_idx})")
            
            tensor_sample = torch.FloatTensor(sample).unsqueeze(0).to(device)
            
            probs = model(tensor_sample, mode='classify')
            conf, pred_idx = torch.max(probs, 1)
            pred_idx = pred_idx.item()
            conf = conf.item() * 100
            
            pred_label_str = idx_to_label.get(pred_idx, f"Unk({pred_idx})")
            
            is_correct = (pred_idx == true_label_idx)
            status = "CORRECT" if is_correct else "WRONG"
            if is_correct:
                correct_count += 1
                
            print(f"{idx:<6} | {true_label_str:<12} | {pred_label_str:<12} | {conf:.1f}% | {status:<10}")
            
    print("-" * 60)
    print(f"Accuracy on random subset: {correct_count}/50 ({correct_count/50*100:.1f}%)")

if __name__ == "__main__":
    test_random_samples_2018()
