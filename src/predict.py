"""
Tahmin Modulu
-------------
Bu modul, egitilmis modeli yukleyip tek bir goruntu uzerinde tahmin yapar.
Streamlit arayuzu bu modulun fonksiyonlarini kullanir.

Temel kullanim:
    from predict import load_model, predict_image
    model, class_names = load_model()
    sonuc = predict_image(model, "path/to/image.jpg", class_names)
"""

from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

import config


def load_model(checkpoint_path: Union[str, Path] = None):
    """
    Egitilmis modeli ve sinif isimlerini yukler.
    
    Args:
        checkpoint_path: Model dosyasinin yolu. None ise config'tekini kullanir.
    
    Returns:
        (model, class_names): Yuklenmis model ve sinif isimleri listesi
    """
    if checkpoint_path is None:
        checkpoint_path = config.MODEL_SAVE_PATH

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model dosyasi bulunamadi: {checkpoint_path}\n"
            f"Once 'python src/train.py' ile modeli egitin."
        )

    # Checkpoint yukle
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE,
                             weights_only=False)

    # Modeli olustur (egitimde kullanilan ayni mimari)
    model_name = checkpoint.get("model_name", config.MODEL_NAME)
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = timm.create_model(
        model_name,
        pretrained=False,  # Pretrained gerekmiyor, kendi agirliklarimizi yukleyecegiz
        num_classes=num_classes,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()

    return model, class_names


def get_inference_transform():
    """
    Tahmin icin goruntu donusumu.
    Egitimdeki val/test transformasyonu ile birebir AYNI olmali.
    """
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


@torch.no_grad()
def predict_image(model, image_input, class_names):
    """
    Tek bir goruntu icin tahmin yapar.
    
    Args:
        model: Yuklenmis PyTorch modeli
        image_input: Goruntu yolu (str/Path) VEYA PIL.Image nesnesi
        class_names: Sinif isimleri listesi
    
    Returns:
        dict: {
            "predicted_class": "SUV",
            "predicted_index": 6,
            "confidence": 0.87,  # En yuksek olasilik
            "probabilities": {  # Tum siniflar icin olasiliklar
                "ACIK_TEKERLEKLI": 0.01,
                "HATCHBACK": 0.05,
                ...
            }
        }
    """
    # Goruntu yukleme
    if isinstance(image_input, (str, Path)):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError(f"Desteklenmeyen goruntu tipi: {type(image_input)}")

    # Donusumu uygula ve batch boyutu ekle (1, C, H, W)
    transform = get_inference_transform()
    tensor = transform(image).unsqueeze(0).to(config.DEVICE)

    # Tahmin
    outputs = model(tensor)
    probabilities = F.softmax(outputs, dim=1)[0]  # (num_classes,)

    # En yuksek olasilikli sinif
    confidence, predicted_idx = probabilities.max(0)
    predicted_idx = predicted_idx.item()
    confidence = confidence.item()

    # Tum siniflar icin olasilik sozlugu
    prob_dict = {
        class_names[i]: probabilities[i].item()
        for i in range(len(class_names))
    }

    return {
        "predicted_class": class_names[predicted_idx],
        "predicted_index": predicted_idx,
        "confidence": confidence,
        "probabilities": prob_dict,
    }


def predict_with_display_name(model, image_input, class_names):
    """
    predict_image'in uzerine, display_name (Turkce guzel isim) de ekler.
    Arayuz icin daha kullanisli.
    """
    result = predict_image(model, image_input, class_names)
    result["display_name"] = config.CLASS_DISPLAY_NAMES.get(
        result["predicted_class"],
        result["predicted_class"]
    )
    return result


if __name__ == "__main__":
    """Komut satirindan test: python src/predict.py <goruntu_yolu>"""
    import sys

    if len(sys.argv) < 2:
        print("Kullanim: python src/predict.py <goruntu_yolu>")
        sys.exit(1)

    image_path = sys.argv[1]

    print(f"Model yukleniyor...")
    model, class_names = load_model()
    print(f"Model yuklendi. Cihaz: {config.DEVICE}")

    print(f"\nGoruntu: {image_path}")
    result = predict_with_display_name(model, image_path, class_names)

    print(f"\n=== TAHMIN SONUCU ===")
    print(f"Tahmin edilen sinif: {result['display_name']}")
    print(f"Guven skoru        : {result['confidence']*100:.2f}%")
    print(f"\nTum siniflar icin olasiliklar:")
    sorted_probs = sorted(result["probabilities"].items(),
                          key=lambda x: x[1], reverse=True)
    for class_name, prob in sorted_probs:
        bar = "#" * int(prob * 40)
        print(f"  {class_name:20s} {prob*100:6.2f}% {bar}")
