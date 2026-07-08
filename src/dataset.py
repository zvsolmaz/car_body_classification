import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import config


def get_train_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=20),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_dataloaders():
    train_dataset = datasets.ImageFolder(
        root=str(config.TRAIN_DIR),
        transform=get_train_transforms()
    )
    val_dataset = datasets.ImageFolder(
        root=str(config.VAL_DIR),
        transform=get_val_transforms()
    )
    test_dataset = datasets.ImageFolder(
        root=str(config.TEST_DIR),
        transform=get_val_transforms()
    )

    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == "cuda" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == "cuda" else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == "cuda" else False,
    )

    return train_loader, val_loader, test_loader, class_names


def check_dataset():
    print("=" * 60)
    print("VERI SETI KONTROLU")
    print("=" * 60)

    for split_name, split_dir in [("TRAIN", config.TRAIN_DIR),
                                   ("VAL", config.VAL_DIR),
                                   ("TEST", config.TEST_DIR)]:
        print(f"\n[{split_name}] - {split_dir}")
        if not split_dir.exists():
            print(f"  HATA: Klasor bulunamadi!")
            continue

        toplam = 0
        for class_name in sorted(config.CLASS_NAMES):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"  {class_name:20s}: KLASOR YOK")
                continue
            uzantilar = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
            sayi = sum(len(list(class_dir.glob(u))) for u in uzantilar)
            print(f"  {class_name:20s}: {sayi:4d} goruntu")
            toplam += sayi
        print(f"  {'TOPLAM':20s}: {toplam:4d} goruntu")

    print("=" * 60)


if __name__ == "__main__":
    check_dataset()
