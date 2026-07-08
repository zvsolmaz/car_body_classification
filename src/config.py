from pathlib import Path
import torch
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
MODEL_SAVE_PATH = MODELS_DIR / "best_model.pth"

CLASS_NAMES = [
    "ACIK_TEKERLEKLI",
    "HATCHBACK",
    "MICRO",
    "PICK_UP",
    "SEDAN",
    "STATION_WAGON",
    "SUV",
    "VAN",
]

CLASS_DISPLAY_NAMES = {
    "ACIK_TEKERLEKLI": "Acik Tekerlekli (F1)",
    "HATCHBACK": "Hatchback",
    "MICRO": "Micro",
    "PICK_UP": "Pick-Up",
    "SEDAN": "Sedan",
    "STATION_WAGON": "Station Wagon",
    "SUV": "SUV",
    "VAN": "Van",
}

NUM_CLASSES = len(CLASS_NAMES)

CLASS_TO_CODE = {
    "SUV": 1,
    "VAN": 2,
    "STATION_WAGON": 3,
    "MICRO": 4,
    "ACIK_TEKERLEKLI": 5,
    "SEDAN": 6,
    "HATCHBACK": 7,
    "PICK_UP": 8,
}

CODE_TO_CLASS = {v: k for k, v in CLASS_TO_CODE.items()}

MODEL_NAME = "efficientnet_b2"
IMAGE_SIZE = 260

BATCH_SIZE = 16
NUM_EPOCHS = 25              # 50'den 25'e indirildi (CPU için)
LEARNING_RATE = 3e-4
EARLY_STOPPING_PATIENCE = 7  # 10'dan 7'ye indirildi
DROPOUT_RATE = 0.4
NUM_WORKERS = 0

LABEL_SMOOTHING = 0.1

# Focal Loss parametresi
# gamma=2: Standart değer, HATCHBACK/MICRO karışıklığı için etkili
FOCAL_GAMMA = 2.0

# Mixup Augmentation parametresi
# alpha=0.3: Düşük karıştırma oranı, çok agresif olmasın
MIXUP_ALPHA = 0.3

PHASE1_EPOCHS = 5
PHASE1_LR = 1e-3
PHASE2_LR = 3e-4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42


def print_config():
    print("=" * 60)
    print("PROJE YAPILANDIRMASI")
    print("=" * 60)
    print(f"Model            : {MODEL_NAME}")
    print(f"Goruntu boyutu   : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Sinif sayisi     : {NUM_CLASSES}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"Max epoch        : {NUM_EPOCHS}")
    print(f"Phase1 LR        : {PHASE1_LR} ({PHASE1_EPOCHS} epoch)")
    print(f"Phase2 LR        : {PHASE2_LR}")
    print(f"Dropout rate     : {DROPOUT_RATE}")
    print(f"Label smoothing  : {LABEL_SMOOTHING}")
    print(f"Focal Loss gamma : {FOCAL_GAMMA}")
    print(f"Mixup alpha      : {MIXUP_ALPHA}")
    print(f"Early stopping   : {EARLY_STOPPING_PATIENCE} epoch")
    print(f"Cihaz            : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()