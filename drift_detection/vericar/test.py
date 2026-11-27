import torch
import yaml
from pathlib import Path
import tqdm
from pytorch_lightning import Trainer, seed_everything

from model import VehiInfoRet
from dataloader2 import VehicleDataModule

def build_exemplar_set(model, dataloader, device):
    print("\nBuilding exemplar set for retrieval testing...")
    model.eval()
    train_loader = dataloader.train_dataloader()

    features_by_class = {}

    with torch.no_grad():
        for batch in tqdm.tqdm(train_loader, desc="Building exemplar set"):
            images, all_labels, _ = batch
            images = images.to(device)
            features = model(images)

            for feature, label in zip(features, all_labels):
                label_item = label.item()
                if label_item not in features_by_class:
                    features_by_class[label_item] = []
                features_by_class[label_item].append(feature)

    exemplar_features = []
    exemplar_labels = []

    for class_idx, class_features_list in features_by_class.items():
        if not class_features_list:
            continue

        class_features = torch.stack(class_features_list)
        if model.exemplar_k >= len(class_features):
            exemplar_features.append(class_features)
            exemplar_labels.extend([class_idx] * len(class_features))
        else:
            mean_feature = torch.mean(class_features, dim=0)
            distances = torch.cdist(mean_feature.unsqueeze(0), class_features).squeeze(0)
            _, top_k_indices = torch.topk(distances, model.exemplar_k, largest=False)
            exemplar_features.append(class_features[top_k_indices])
            exemplar_labels.extend([class_idx] * model.exemplar_k)

    if exemplar_features:
        model.exemplar_features = torch.cat(exemplar_features, dim=0).to(device)
        model.exemplar_labels = torch.tensor(exemplar_labels, device=device, dtype=torch.long)
        print(f"Exemplar set built with {len(model.exemplar_features)} samples.")
    else:
        print("Warning: Could not build exemplar set.")

def test_config(config, ckpt_path=None):
    seed_everything(config.get('seed', 42), workers=True)

    datamodule = VehicleDataModule(
        data_root=config['data']['data_root'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        flip_p=0.0,
    )
    datamodule.setup(stage=None)
    test_loader = datamodule.test_dataloader()

    if ckpt_path is None:
        ckpt_path = config['test']['ckpt_path']
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from checkpoint: {ckpt_path} (map_location={device})")
    model = VehiInfoRet.load_from_checkpoint(ckpt_path, map_location=device)

    new_exemplar_k = config['training'].get('exemplar_k', 15)
    print(f"Overwriting exemplar_k: from {model.exemplar_k} (in ckpt) to {new_exemplar_k} (in config.yaml)")
    model.exemplar_k = new_exemplar_k
    
    model.to(device)
    model.eval()

    build_exemplar_set(model, datamodule, device)

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=[0] if torch.cuda.is_available() else 'auto',
        logger=False # Disable logging for pure testing
    )

    print("\nStarting model testing...")
    test_results = trainer.test(model, dataloaders=test_loader, verbose=True)
    print("Testing finished.")
    
    print("\n--- Test Results ---")
    if test_results:
        precision_1 = test_results[0].get('test_prec1', 0.0)
        map_r = test_results[0].get('test_map_r', 0.0)
        print(f"Precision@1: {precision_1:.4f}")
        print(f"mAP@R: {map_r:.4f}")
        return {'prec1': precision_1, 'map_r': map_r}
    else:
        print("Could not retrieve test results.")
        return {'prec1': 0.0, 'map_r': 0.0}

if __name__ == '__main__':
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please create it.")
        exit()

    if 'test' not in config or 'ckpt_path' not in config['test']:
        print("Error: 'testing' section with 'ckpt_path' not found in config.yaml.")
        exit()

    test_config(config)
