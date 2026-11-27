import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EmpiricalCovariance
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

def get_mahalanobis_params(features, labels):
    unique_labels = torch.unique(labels)
    class_means = []
    centered_features = []
    
    print(f"Computing Mahalanobis params on {len(features)} samples, {len(unique_labels)} classes...")
    
    for label in unique_labels:
        class_feats = features[labels == label]
        mean = torch.mean(class_feats, dim=0)
        class_means.append(mean)
        centered_features.append(class_feats - mean)
        
    class_means = torch.stack(class_means) # [C, D]
    all_centered = torch.cat(centered_features, dim=0).numpy()
    
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(all_centered)
    
    precision = torch.from_numpy(ec.precision_).float()
    
    return class_means, precision

def calculate_mahalanobis_scores(features, class_means, precision, device):
    features = features.to(device)
    class_means = class_means.to(device)
    precision = precision.to(device)
    
    
    XP = torch.matmul(features, precision)
    term1 = torch.sum(XP * features, dim=1)
    
    MuP = torch.matmul(class_means, precision)
    term3 = torch.sum(MuP * class_means, dim=1)
    
    term2 = -2 * torch.matmul(XP, class_means.t())
    
    dists = term1.unsqueeze(1) + term2 + term3.unsqueeze(0)
    
    min_dist, _ = torch.min(dists, dim=1)
    
    return min_dist.cpu()

def ood_detection_mahalanobis_main(config, ckpt_path=None):
    pl.seed_everything(config.get('seed', 42), workers=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if ckpt_path is None:
        ckpt_path = config['step1']['ckpt_path']
        
    print(f"\n[Mahalanobis] Loading checkpoint: {ckpt_path} (map_location={device})")
    model = VehiInfoRet.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    
    print("[Mahalanobis] Loading Step 0 Train Data...")
    step0_dm = VehicleDataModule(
        data_root=config['data']['data_root'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        flip_p=0.0
    )
    step0_dm.setup(stage='fit')
    train_loader = step0_dm.train_dataloader()
    
    train_feats, train_labels = extract_features(model, train_loader, device)
    class_means, precision = get_mahalanobis_params(train_feats, train_labels)
    
    print("[Mahalanobis] Loading Step 0 Test (ID) and Step 1 (OOD)...")
    step0_dm_test = VehicleDataModule(
        data_root=config['data']['data_root'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        flip_p=0.0
    )
    step0_dm_test.setup(stage='test')
    test_loader_id = step0_dm_test.test_dataloader()
    
    step1_dm = VehicleDataModule(
        data_root=config['test']['data_root'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        flip_p=0.0
    )
    step1_dm.setup(stage='fit')
    loader_ood = step1_dm.train_dataloader()
    
    id_features, _ = extract_features(model, test_loader_id, device)
    ood_features, _ = extract_features(model, loader_ood, device)
    
    print("[Mahalanobis] Calculating scores...")
    id_dists = calculate_mahalanobis_scores(id_features, class_means, precision, device)
    ood_dists = calculate_mahalanobis_scores(ood_features, class_means, precision, device)
    
    id_scores = -id_dists.numpy()
    ood_scores = -ood_dists.numpy()
    
    X = np.concatenate([id_scores, ood_scores])
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    
    auroc = roc_auc_score(y_true, X)
    
    threshold_95 = np.percentile(id_scores, 5)
    fp_count = np.sum(ood_scores > threshold_95)
    fpr95 = fp_count / len(ood_scores)
    
    print(f"[Mahalanobis Results] AUROC: {auroc:.4f}, FPR95: {fpr95:.4f}")
    
    return {
        "auroc": auroc,
        "fpr95": fpr95
    }
