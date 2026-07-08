from PIL import Image
from pathlib import Path

dataset_dir = Path(r"C:\Users\zeynep\Downloads\araba_govde_siniflandirma\araba_govde_siniflandirma\dataset")

bozuk = []
for img_path in dataset_dir.rglob("*.jpg"):
    try:
        with Image.open(img_path) as img:
            img.verify()
    except Exception as e:
        bozuk.append(img_path)
        print(f"Bozuk: {img_path}")

print(f"\nToplam bozuk: {len(bozuk)}")
for p in bozuk:
    p.unlink()
print("Silindi!")