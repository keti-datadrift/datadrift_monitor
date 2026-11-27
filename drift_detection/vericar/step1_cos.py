import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl

# Import your modules
from dataloader2 import VehicleDataModule 
from model import VehiInfoRet

def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images, all_labels, _ = batch
            images = images.to(device)
            
            features = model(images)
            
            features_list.append(features.cpu())
            labels_list.append(all_labels.cpu())

    if len(features_list) == 0:
        return torch.tensor([]), torch.tensor([])
        
    return torch.cat(features_list), torch.cat(labels_list)

def calculate_ood_metrics(id_features, ood_features, exemplar_features, k=5):

    gallery = exemplar_features.cpu()
    
    X = torch.cat([id_features, ood_features], dim=0)

    y_true = np.concatenate([np.ones(len(id_features)), np.zeros(len(ood_features))])
    
    print(f"Calculating cosine similarities for {len(X)} queries against {len(gallery)} exemplars...")
    
    sims = torch.matmul(X, gallery.T)
    
    topk_sims, _ = torch.topk(sims, k=k, dim=1, largest=True)
    
    mean_sims = topk_sims.mean(dim=1).numpy()
        
    id_sims = mean_sims[:len(id_features)]
    ood_sims = mean_sims[len(id_features):]
    
    threshold_95 = np.percentile(id_sims, 5)
    
    fp_count = np.sum(ood_sims > threshold_95)
    fpr_at_95 = fp_count / len(ood_sims)
    
    scores = mean_sims
    auroc = roc_auc_score(y_true, scores)
    
    print(f"\n[Analysis Results]")
    print(f"Threshold (at 95% ID Recall): {threshold_95:.4f}")
    print(f"Mean Sim ID : {id_sims.mean():.4f}")
    print(f"Mean Sim OOD: {ood_sims.mean():.4f}")

    return auroc, fpr_at_95, threshold_95

def evaluate_retrieval_performance(model, features, labels, description, device):
    if len(features) == 0:
        print(f"\n--- Retrieval Performance ({description}) ---")
        print("No data to evaluate.")
        return 0.0, 0.0

    features = features.to(device)
    labels = labels.to(device)

    prec1, map_r = model.calc_retrieval_metrics(features, labels)
    prec1_val = prec1.item()
    map_r_val = map_r.item()
    print(f"\n--- Retrieval Performance ({description}) ---")
    print(f"Precision@1: {prec1_val:.4f}")
    print(f"mAP@R      : {map_r_val:.4f}")
    return prec1_val, map_r_val

def sample_exemplars(features, labels, exemplar_k):
    unique_labels = torch.unique(labels)
    new_exemplar_features = []
    new_exemplar_labels = []

    print(f"Sampling {exemplar_k} exemplars per class...")

    for label in unique_labels:
        indices = (labels == label).nonzero(as_tuple=True)[0]
        class_features = features[indices]
        
        class_mean = class_features.mean(dim=0, keepdim=True)
        class_mean = F.normalize(class_mean, p=2, dim=1)
        
        sims = torch.matmul(class_features, class_mean.t()).squeeze(1)
        
        k = min(len(indices), exemplar_k)
        _, topk_indices = torch.topk(sims, k=k, largest=True)
        
        selected_features = class_features[topk_indices]
        
        new_exemplar_features.append(selected_features)
        new_exemplar_labels.append(torch.full((k,), label, dtype=torch.long))

    return torch.cat(new_exemplar_features), torch.cat(new_exemplar_labels)

def ood_detection_main(config, ckpt_path=None):
    pl.seed_everything(config.get('seed', 42), workers=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if ckpt_path is None:
        ckpt_path = config['step1']['ckpt_path']
        
    print(f"Loading checkpoint from: {ckpt_path} (map_location={device})")
    model = VehiInfoRet.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()

    if getattr(model, 'exemplar_features', None) is None:
        print("\n[Warning] Exemplars not found in checkpoint. Regenerating from Step 0 Train Data...")
        
        step0_dm = VehicleDataModule(
            data_root=config['data']['data_root'],
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            flip_p=0.0,
        )
        step0_dm.setup(stage='fit')
        train_loader_step0 = step0_dm.train_dataloader()
        
        print("Extracting features from Step 0 Train Set...")
        train_feats, train_labels = extract_features(model, train_loader_step0, device)
        
        ex_k = config['training']['exemplar_k']
        ex_feats, ex_labels = sample_exemplars(train_feats, train_labels, ex_k)
        
        if hasattr(model, 'exemplar_features'):
            del model.exemplar_features
        if hasattr(model, 'exemplar_labels'):
            del model.exemplar_labels

        model.register_buffer('exemplar_features', ex_feats.to(device))
        model.register_buffer('exemplar_labels', ex_labels.to(device))
        
        print(f"Regenerated {len(ex_feats)} exemplars from Step 0.")
    else:
        print("Exemplars found in checkpoint.")
    
    print("\nSetting up Step 0 Data (Test Set)...")
    step0_dm_test = VehicleDataModule(
        data_root=config['data']['data_root'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        flip_p=0.0,
    )
    step0_dm_test.setup(stage='test')
    test_loader_step0 = step0_dm_test.test_dataloader()
    
    print("Setting up Step 1 Data (OOD Set)...")
    step1_dm = VehicleDataModule(
        data_root=config['test']['data_root'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        flip_p=0.0,
    )
    step1_dm.setup(stage='fit') 
    loader_step1 = step1_dm.train_dataloader()

    print("Extracting features for Step 0 Test (ID)...")
    id_features, id_labels = extract_features(model, test_loader_step0, device)
    
    print("Extracting features for Step 1 (OOD)...")
    ood_features, ood_labels = extract_features(model, loader_step1, device)

    ood_k = config['test']['ood_k']
    current_exemplars = model.exemplar_features
    
    auroc, fpr95, threshold = calculate_ood_metrics(id_features, ood_features, current_exemplars, k=ood_k)
    
    print("="*40)
    print(f"OOD Detection Results (k={ood_k})")
    print(f"AUROC: {auroc:.4f}")
    print(f"FPR95: {fpr95:.4f} (at TPR=95%, Threshold={threshold:.4f})")
    print("="*40)

    exemplar_k = config['training']['exemplar_k']
    
    step1_ex_features, step1_ex_labels = sample_exemplars(ood_features, ood_labels, exemplar_k)
    
    print(f"Selected {len(step1_ex_features)} new exemplars from Step 1.")
    
    step1_ex_features = step1_ex_features.to(device)
    step1_ex_labels = step1_ex_labels.to(device)
    
    updated_features = torch.cat([model.exemplar_features, step1_ex_features], dim=0)
    updated_labels = torch.cat([model.exemplar_labels, step1_ex_labels], dim=0)
    
    model.exemplar_features = updated_features
    model.exemplar_labels = updated_labels
    
    print("\n" + "="*40)
    print("Evaluating retrieval performance with updated exemplars...")

    combined_features = torch.cat([id_features, ood_features], dim=0)
    combined_labels = torch.cat([id_labels, ood_labels], dim=0)
    comb_p1, comb_map = evaluate_retrieval_performance(model, combined_features, combined_labels, "Combined (181 classes)", device)

    seen_p1, seen_map = evaluate_retrieval_performance(model, id_features, id_labels, "Seen (161 classes)", device)

    unseen_p1, unseen_map = evaluate_retrieval_performance(model, ood_features, ood_labels, "Unseen (20 classes)", device)
    print("="*40)

    return {
        "auroc": auroc,
        "fpr95": fpr95,
        "combined_prec1": comb_p1,
        "combined_map": comb_map,
        "seen_prec1": seen_p1,
        "seen_map": seen_map,
        "unseen_prec1": unseen_p1,
        "unseen_map": unseen_map
    }

if __name__ == '__main__':
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please create it.")
        exit()
    
    ood_detection_main(config)
