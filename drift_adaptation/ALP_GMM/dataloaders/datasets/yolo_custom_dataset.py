import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset



class YoloCustomDataset(Dataset):
    def __init__(self, df, img_size=640):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path, label_path = row["image_path"], row["label_path"]

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Load YOLO labels
        with open(label_path, "r") as f:
            labels = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x, y, w, h = map(float, parts)
                    labels.append([cls, x, y, w, h])
        
        if len(labels) == 0:
            labels = torch.zeros((0, 5), dtype=torch.float32)
        else:
            labels = torch.tensor(labels, dtype=torch.float32)

        return {
            "img": img,  # (C, H, W)
            "bboxes": labels[:, 1:],
            "cls": labels[:, 0].unsqueeze(-1),
            "im_file": img_path
        }


def yolo_collate_fn(batch):
    imgs = []
    bboxes = []
    cls = []
    batch_idx = []
    im_files = []

    for i, sample in enumerate(batch):
        imgs.append(sample["img"])
        im_files.append(sample["im_file"])
        if sample["bboxes"].numel() > 0:
            bboxes.append(sample["bboxes"])
            cls.append(sample["cls"])
            batch_idx.append(torch.full((len(sample["bboxes"]),), i))

    imgs = torch.stack(imgs, dim=0)

    if len(bboxes) > 0:
        bboxes = torch.cat(bboxes, dim=0)
        cls = torch.cat(cls, dim=0)
        batch_idx = torch.cat(batch_idx, dim=0)
    else:
        bboxes = torch.zeros((0, 4), dtype=torch.float32)
        cls = torch.zeros((0, 1), dtype=torch.float32)
        batch_idx = torch.zeros((0,), dtype=torch.int64)

    return {
        "img": imgs,
        "bboxes": bboxes,
        "cls": cls,
        "batch_idx": batch_idx,
        "im_file": im_files
    }