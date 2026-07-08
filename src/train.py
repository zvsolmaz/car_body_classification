import time
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import config
from dataset import get_dataloaders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── FOCAL LOSS ────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss: Zor örneklere (yanlış sınıflandırılan) daha fazla odaklanır.
    gamma=2: Kolay örneklerin katkısını azaltır, HATCHBACK/MICRO gibi
             karışan sınıfların öğrenimini güçlendirir.
    alpha (class weights): Az veri olan sınıflara daha fazla ağırlık verir.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha      # class weights tensoru
        self.gamma = gamma      # odaklanma parametresi (2.0 standart değer)
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Önce standart cross entropy hesapla
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        # Softmax olasılıklarından doğru sınıfın olasılığını al
        pt = torch.exp(-ce_loss)
        # Focal ağırlığı uygula: (1 - pt)^gamma
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


# ─── MIXUP ────────────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.3, device='cpu'):
    """
    Mixup Augmentation: İki farklı görseli ve etiketini karıştırır.
    alpha=0.3: Karıştırma oranı (düşük tutuldu, çok agresif olmasın).
    HATCHBACK/MICRO sınır bölgelerini öğrenmek için etkili.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup için kayıp: iki etiketin ağırlıklı ortalaması."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─── YARDIMCI FONKSİYONLAR ────────────────────────────────────────────────
def compute_class_weights(train_dir, class_names):
    """
    Sınıf bazında görsel sayısına göre ağırlık hesaplar.
    Az görsel olan sınıflara daha yüksek ağırlık verir.
    """
    counts = []
    for cls in class_names:
        cls_dir = Path(train_dir) / cls
        n = len(list(cls_dir.glob("*.jpg"))) + \
            len(list(cls_dir.glob("*.jpeg"))) + \
            len(list(cls_dir.glob("*.png")))
        counts.append(max(n, 1))

    counts = np.array(counts, dtype=float)
    weights = counts.max() / counts
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(config.DEVICE)

    print("\nClass Weights:")
    for cls, w, c in zip(class_names, weights, counts):
        bar = "█" * int(w * 10)
        print(f"  {cls:20s}: {w:.3f}  (n={int(c)})  {bar}")
    return weights_tensor


def build_model(freeze_backbone=False):
    model = timm.create_model(
        config.MODEL_NAME,
        pretrained=True,
        num_classes=config.NUM_CLASSES,
        drop_rate=config.DROPOUT_RATE,
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        print("Backbone donduruldu - sadece classifier egitiliyor")
    else:
        for param in model.parameters():
            param.requires_grad = True
        print("Tum model egitiliyor (fine-tune)")

    model = model.to(config.DEVICE)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    """
    use_mixup=True: Eğitimde Mixup augmentation kullan.
    Sadece eğitimde uygulanır, validasyon/test'te kullanılmaz.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup and random.random() > 0.5:
            # %50 ihtimalle Mixup uygula (her batch'te değil)
            mixed_images, y_a, y_b, lam = mixup_data(
                images, labels, alpha=config.MIXUP_ALPHA, device=device)
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / total_samples,
            total_correct / total_samples,
            np.array(all_labels),
            np.array(all_preds))


def run_training_phase(model, train_loader, val_loader,
                       criterion, lr, num_epochs, phase_name,
                       history, best_val_loss, best_model_state,
                       use_mixup=True):
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n--- [{phase_name}] Epoch {epoch}/{num_epochs} ---")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE,
            use_mixup=use_mixup)
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, config.DEVICE)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc*100:.2f}%")
        print(f"  LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_model_state,
                "class_names": train_loader.dataset.classes,
                "model_name": config.MODEL_NAME,
                "image_size": config.IMAGE_SIZE,
            }, config.MODEL_SAVE_PATH)
            print(f"  En iyi model guncellendi (val_loss={val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Iyilesme yok ({epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE})")
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarlyStopping: {phase_name} erken sonlandirildi.")
                break

    return best_val_loss, best_model_state


# ─── GRAFİK FONKSİYONLARI ──────────────────────────────────────────────────

def plot_curves(history, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss", linewidth=2)
    plt.plot(history["val_loss"],   label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_plot.png", dpi=150)
    plt.close()
    print("Grafik kaydedildi: loss_plot.png")

    plt.figure(figsize=(10, 6))
    plt.plot([a * 100 for a in history["train_acc"]], label="Training Accuracy", linewidth=2)
    plt.plot([a * 100 for a in history["val_acc"]],   label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_plot.png", dpi=150)
    plt.close()
    print("Grafik kaydedildi: accuracy_plot.png")


def plot_confusion_matrix_normalized(y_true, y_pred, class_names, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Oran"}, vmin=0, vmax=1,
    )
    plt.xlabel("Tahmin Edilen Sinif")
    plt.ylabel("Gercek Sinif")
    plt.title("Normalize Edilmis Confusion Matrix (Oranlar)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix_normalized.png", dpi=150)
    plt.close()
    print("Grafik kaydedildi: confusion_matrix_normalized.png")


def plot_confusion_matrix_raw(y_true, y_pred, class_names, save_dir):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"label": "Goruntu Sayisi"},
    )
    plt.xlabel("Tahmin Edilen Sinif")
    plt.ylabel("Gercek Sinif")
    plt.title("Confusion Matrix (Ham Sayilar)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix_raw.png", dpi=150)
    plt.close()
    print("Grafik kaydedildi: confusion_matrix_raw.png")


def plot_per_class_accuracy(y_true, y_pred, class_names, save_dir):
    per_class_acc = []
    for i in range(len(class_names)):
        mask = y_true == i
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            per_class_acc.append((y_pred[mask] == y_true[mask]).mean())

    colors = [
        "#2ecc71" if a >= 0.80 else
        "#f39c12" if a >= 0.60 else
        "#e74c3c"
        for a in per_class_acc
    ]

    plt.figure(figsize=(13, 6))
    bars = plt.bar(class_names, [a * 100 for a in per_class_acc],
                   color=colors, edgecolor="white", linewidth=0.8)

    for bar, acc in zip(bars, per_class_acc):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{acc:.1%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    plt.axhline(80, color="green",  linestyle="--", alpha=0.6, label="Hedef %80")
    plt.axhline(60, color="orange", linestyle="--", alpha=0.6, label="Kritik %60")
    plt.ylim(0, 115)
    plt.title("Sinif Bazinda Test Dogruluk Orani")
    plt.ylabel("Dogruluk (%)")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="≥ %80 (iyi)"),
        Patch(facecolor="#f39c12", label="%60-%80 (orta)"),
        Patch(facecolor="#e74c3c", label="< %60 (zayif)"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_dir / "per_class_accuracy.png", dpi=150)
    plt.close()
    print("Grafik kaydedildi: per_class_accuracy.png")

    print("\nSinif Bazinda Dogruluk:")
    for cls, acc in zip(class_names, per_class_acc):
        status = "✅" if acc >= 0.80 else "⚠️" if acc >= 0.60 else "❌"
        bar = "█" * int(acc * 20)
        print(f"  {status} {cls:20s}: {acc:.1%}  {bar}")


def plot_per_class_metrics(y_true, y_pred, class_names, save_dir):
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    precisions = [report[c]["precision"] for c in class_names]
    recalls    = [report[c]["recall"]    for c in class_names]
    f1s        = [report[c]["f1-score"]  for c in class_names]

    x = np.arange(len(class_names))
    w = 0.25

    plt.figure(figsize=(14, 6))
    plt.bar(x - w, precisions, w, label="Precision", color="#3498db", alpha=0.85)
    plt.bar(x,     recalls,    w, label="Recall",    color="#2ecc71", alpha=0.85)
    plt.bar(x + w, f1s,        w, label="F1-Score",  color="#e74c3c", alpha=0.85)

    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0, 1.1)
    plt.ylabel("Skor")
    plt.title("Sinif Bazinda Precision / Recall / F1-Score")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_dir / "per_class_metrics.png", dpi=150)
    plt.close()
    print("Grafik kaydedildi: per_class_metrics.png")


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    set_seed(config.RANDOM_SEED)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IKI ASAMALI EGITIM BASLIYOR")
    print(f"Focal Loss gamma : {config.FOCAL_GAMMA}")
    print(f"Mixup alpha      : {config.MIXUP_ALPHA}")
    print("=" * 60)

    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print(f"Train batch: {len(train_loader)} | Val batch: {len(val_loader)}")
    print(f"Siniflar: {class_names}")

    if len(train_loader) == 0:
        print("HATA: Egitim verisi bulunamadi!")
        return

    # Class weights hesapla
    class_weights = compute_class_weights(config.TRAIN_DIR, class_names)

    # Focal Loss (class weights + label smoothing dahil)
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=config.FOCAL_GAMMA,
        label_smoothing=config.LABEL_SMOOTHING
    )

    history = {"train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_model_state = None

    start = time.time()

    # ── ASAMA 1: Sadece classifier (Mixup yok) ──────────────
    print("\n" + "=" * 60)
    print(f"ASAMA 1: Classifier Egitimi ({config.PHASE1_EPOCHS} epoch, LR={config.PHASE1_LR})")
    print("=" * 60)

    model = build_model(freeze_backbone=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Egitilebilir parametre: {n_params:,}")

    best_val_loss, best_model_state = run_training_phase(
        model, train_loader, val_loader,
        criterion, config.PHASE1_LR,
        config.PHASE1_EPOCHS, "ASAMA-1",
        history, best_val_loss, best_model_state,
        use_mixup=False   # Asama 1'de Mixup kapalı
    )

    # ── ASAMA 2: Fine-tune (Mixup açık) ─────────────────────
    print("\n" + "=" * 60)
    print(f"ASAMA 2: Fine-tuning (max {config.NUM_EPOCHS} epoch, LR={config.PHASE2_LR})")
    print("=" * 60)

    if best_model_state:
        model.load_state_dict(best_model_state)

    for param in model.parameters():
        param.requires_grad = True

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Egitilebilir parametre: {n_params:,}")

    best_val_loss, best_model_state = run_training_phase(
        model, train_loader, val_loader,
        criterion, config.PHASE2_LR,
        config.NUM_EPOCHS, "ASAMA-2",
        history, best_val_loss, best_model_state,
        use_mixup=True    # Asama 2'de Mixup açık
    )

    elapsed = time.time() - start
    print(f"\nToplam egitim suresi: {elapsed/60:.2f} dakika")

    if best_model_state:
        model.load_state_dict(best_model_state)

    # ── GRAFİKLER ────────────────────────────────────────────
    print("\n=== GRAFIKLER ===")
    plot_curves(history, config.RESULTS_DIR)

    # ── TEST DEĞERLENDİRMESİ ─────────────────────────────────
    print("\n=== TEST SETI ===")
    if len(test_loader) > 0:
        test_loss, test_acc, y_true, y_pred = evaluate(
            model, test_loader, criterion, config.DEVICE)

        print(f"Test Loss    : {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc*100:.2f}%")

        plot_confusion_matrix_normalized(y_true, y_pred, class_names, config.RESULTS_DIR)
        plot_confusion_matrix_raw(y_true, y_pred, class_names, config.RESULTS_DIR)
        plot_per_class_accuracy(y_true, y_pred, class_names, config.RESULTS_DIR)
        plot_per_class_metrics(y_true, y_pred, class_names, config.RESULTS_DIR)

        report = classification_report(
            y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
        print("\nClassification Report:")
        print(report)

        report_path = config.RESULTS_DIR / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n")
            f.write(report)
        print(f"Rapor kaydedildi: {report_path}")

    print("\n" + "=" * 60)
    print("EGITIM TAMAMLANDI")
    print("=" * 60)
    print(f"En iyi model : {config.MODEL_SAVE_PATH}")
    print(f"Sonuclar     : {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()