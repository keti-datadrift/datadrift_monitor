import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import shutil
import tqdm
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from viewpoint_train import ViewpointClassifier


class InferenceDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        valid = True
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 에러 발생 시 더미 텐서 반환 및 valid 플래그 False 설정
            image = torch.zeros(3, 224, 224)
            valid = False

        return image, str(img_path), valid


def main():
    source_dir = Path(r"E:\viewpoint\used")
    dest_root = Path(r"E:\viewpoint")
    ckpt_dir = Path(r"lightning_logs\version_0\checkpoints")

    batch_size = 256
    num_workers = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not ckpt_dir.exists():
        ckpt_dir = Path(__file__).parent / "lightning_logs" / "version_0" / "checkpoints"

    ckpt_files = list(ckpt_dir.glob("*.ckpt"))
    if not ckpt_files:
        print(f"Error: No checkpoint found in {ckpt_dir}")
        return

    ckpt_path = ckpt_files[0]
    print(f"Loading model from: {ckpt_path}")

    model = ViewpointClassifier.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    print(f"Scanning images in {source_dir}...")
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths_set = set()
    for ext in extensions:
        image_paths_set.update(source_dir.rglob(ext))
        image_paths_set.update(source_dir.rglob(ext.upper()))
    image_paths = list(image_paths_set)

    print(f"Found {len(image_paths)} images.")
    if len(image_paths) == 0:
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = InferenceDataset(image_paths, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    front_dir = dest_root / "front"
    rear_dir = dest_root / "rear"
    front_dir.mkdir(parents=True, exist_ok=True)
    rear_dir.mkdir(parents=True, exist_ok=True)

    print("Starting classification and copying...")
    copy_count = {"front": 0, "rear": 0}

    with torch.no_grad():
        for batch_images, batch_paths, batch_valid in tqdm.tqdm(dataloader):
            batch_images = batch_images.to(device)
            logits = model(batch_images)
            preds = torch.argmax(logits, dim=1)

            for pred, src_path_str, is_valid in zip(preds, batch_paths, batch_valid):
                if not is_valid:
                    continue

                src_path = Path(src_path_str)
                label = pred.item()
                target_folder = front_dir if label == 0 else rear_dir
                key = "front" if label == 0 else "rear"
                
                dst_path = target_folder / src_path.name

                if dst_path.exists():
                    stem = src_path.stem
                    suffix = src_path.suffix
                    counter = 1
                    while dst_path.exists():
                        dst_path = target_folder / f"{stem}_{counter}{suffix}"
                        counter += 1
                
                try:
                    shutil.copy2(src_path, dst_path)
                    copy_count[key] += 1
                except Exception as e:
                    print(f"Failed to copy {src_path}: {e}")

    print(f"Classification finished.")
    print(f"Front images copied: {copy_count['front']}")
    print(f"Rear images copied: {copy_count['rear']}")


if __name__ == "__main__":
    main()