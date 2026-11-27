import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode

try:
    from transformers import CLIPImageProcessor
except ImportError:  # pragma: no cover - optional dependency
    CLIPImageProcessor = None

class SubsetImageFolder(datasets.ImageFolder):

    def __init__(self,
                 root,
                 include_folders,
                 transform=None,
                 target_transform=None):

        assert len(include_folders) == len(set(include_folders))

        self.include_folders = include_folders
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)

    def find_classes(self, directory):

        all_classes = [entry.name for entry in os.scandir(directory) if entry.is_dir()]

        if not all_classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        is_subset = set(self.include_folders).issubset(all_classes)
        if not is_subset:
            raise ValueError("Specified classes must be a subset of existing classes.")
        
        classes = sorted(cls for cls in all_classes if cls in self.include_folders)
        
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx 


class CLIPProcessorTransform:
    def __init__(self, processor, augment=None, use_processor_defaults=True):
        self.processor = processor
        self.augment = augment
        self.use_processor_defaults = use_processor_defaults

    def __call__(self, image):
        if self.augment is not None:
            image = self.augment(image)

        if self.use_processor_defaults:
            pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"]
        else:
            pixel_values = self.processor(
                images=image,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
            )["pixel_values"]
        return pixel_values.squeeze(0)


def gen_datasets(datadir,
                 id_subset,
                 ood_subset,
                 mean=None,
                 std=None,
                 resize=None,
                 cropsize=None,
                 preset="imagenet",
                 model_name=None,
                 ):
    train_transform, eval_transform = _build_transforms(
        preset,
        mean=mean,
        std=std,
        resize=resize,
        cropsize=cropsize,
        model_name=model_name,
    )

    train_dataset = SubsetImageFolder(os.path.join(datadir, "train"),
                                      id_subset,
                                      train_transform)
    val_dataset = SubsetImageFolder(os.path.join(datadir, "val"),
                                    id_subset,
                                    eval_transform)
    ood_dataset = SubsetImageFolder(os.path.join(datadir, "val"),
                                    ood_subset,
                                    eval_transform)

    print("# ID Train: {}".format(len(train_dataset.imgs)))
    print("# ID Val: {}".format(len(val_dataset.imgs)))
    print("# OOD: {}".format(len(ood_dataset.imgs)))

              
    return train_dataset, val_dataset, ood_dataset

def gen_custom_dataset(
    datadir,
    name,
    subset,
    evaluate=True,
    mean=None,
    std=None,
    resize=None,
    cropsize=None,
    preset="imagenet",
    model_name=None,
):
    train_transform, eval_transform = _build_transforms(
        preset,
        mean=mean,
        std=std,
        resize=resize,
        cropsize=cropsize,
        model_name=model_name,
    )

    transform = eval_transform if evaluate else train_transform

    dataset = SubsetImageFolder(os.path.join(datadir, name),
                                subset,
                                transform)

    print(f"# Samples {name}: {len(dataset.imgs)}")

    return dataset


def get_id_classes(id_classes_fn):

    with open(id_classes_fn, "r") as f:
        lines = [line.strip() for line in f]

    return sorted(lines)


def _resolve_preprocessing(preset, mean=None, std=None, resize=None, cropsize=None):
    if preset == "clip":
        default_mean = [0.48145466, 0.4578275, 0.40821073]
        default_std = [0.26862954, 0.26130258, 0.27577711]
        default_resize = 224
        default_cropsize = 224
        interpolation = InterpolationMode.BICUBIC
    else:
        default_mean = [0.485, 0.456, 0.406]
        default_std = [0.229, 0.224, 0.225]
        default_resize = 256
        default_cropsize = 224
        interpolation = InterpolationMode.BILINEAR

    return (
        mean if mean is not None else default_mean,
        std if std is not None else default_std,
        resize if resize is not None else default_resize,
        cropsize if cropsize is not None else default_cropsize,
        interpolation,
    )


def _build_transforms(preset, mean=None, std=None, resize=None, cropsize=None, model_name=None):
    mean, std, resize, cropsize, interpolation = _resolve_preprocessing(
        preset,
        mean=mean,
        std=std,
        resize=resize,
        cropsize=cropsize,
    )

    if preset == "clip":
        if CLIPImageProcessor is None:
            raise ImportError(
                "transformers is required for CLIP preprocessing but is not installed."
            )
        processor = CLIPImageProcessor.from_pretrained(
            model_name or "openai/clip-vit-base-patch32"
        )
        train_augment = transforms.Compose([
            transforms.RandomResizedCrop(cropsize, interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
        ])
        train_transform = CLIPProcessorTransform(
            processor,
            augment=train_augment,
            use_processor_defaults=False,
        )
        eval_transform = CLIPProcessorTransform(
            processor,
            augment=None,
            use_processor_defaults=True,
        )
        return train_transform, eval_transform

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(cropsize, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(resize, interpolation=interpolation),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform
