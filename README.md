# Araba Gövde Tipi Sınıflandırma (EfficientNet-B2)

Bu proje, araç görüntülerinden **8 farklı gövde tipini** (SUV, Van, Station Wagon, Micro, Açık Tekerlekli, Sedan, Hatchback, Pick-Up) otomatik olarak sınıflandıran bir derin öğrenme sistemidir. Model, **EfficientNet-B2** mimarisi üzerine çift aşamalı transfer öğrenme stratejisiyle eğitilmiş ve **Streamlit** tabanlı bir web arayüzü ile gerçek zamanlı tahmin yapabilir hale getirilmiştir.

## 📊 Ekran Görüntüleri

### Görüntü Yükleme ve Tahmin Sonucu
![Streamlit Arayüzü - Tahmin](images/streamlit_prediction.png)

### Olasılık Dağılımı ve Güven Skoru
![Olasılık Dağılımı](images/probability_distribution.png)

### Desteklenen Sınıflar Paneli
![Desteklenen Sınıflar](images/supported_classes.png)

---

## 🚀 Özellikler

* **8 Sınıflı Gövde Tipi Tespiti:** SUV, Van, Station Wagon, Micro, Açık Tekerlekli, Sedan, Hatchback ve Pick-Up sınıflarını yüksek doğrulukla ayırt eder.
* **EfficientNet-B2 Tabanlı Mimari:** ~36 MB model boyutu, ImageNet üzerinde önceden eğitilmiş ağırlıklar ve 260×260 piksel giriş çözünürlüğü ile ResNet-50'ye kıyasla daha düşük hesaplama maliyetinde daha yüksek doğruluk sunar.
* **Çift Aşamalı Eğitim Stratejisi:** Aşama 1'de yalnızca sınıflandırıcı başlığı eğitilir (önceden eğitilmiş katmanlar donuk), Aşama 2'de tüm ağ ince ayar yapılır (ReduceLROnPlateau ile adaptif öğrenme oranı azaltma).
* **Focal Loss + Sınıf Ağırlıkları:** Sınıf dengesizliği sorununu gidermek ve görsel açıdan birbirine karışan Hatchback/Micro gibi sınıfların öğrenimini güçlendirmek için kullanılır.
* **Mixup Augmentation:** İki eğitim örneğini ve etiketini doğrusal olarak karıştırarak (β dağılımı, α=0.3) sınıf sınır bölgelerinde daha pürüzsüz bir karar yüzeyi oluşturur; özellikle Hatchback–Sedan ayrımını güçlendirir.
* **Kapsamlı Veri Artırma:** RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation, RandomErasing, RandomVerticalFlip, RandomGrayscale ve RandomPerspective dönüşümleriyle modelin gerçek dünya koşullarına karşı dayanıklılığı artırılmıştır.
* **Hash Tabanlı Veri Seti Temizliği:** Eğitim, doğrulama ve test setleri arasında görsel tekrarı olmadığı hash kontrolü ile doğrulanmıştır.
* **Gerçek Zamanlı Web Arayüzü:** Streamlit tabanlı arayüz; sürükle-bırak görüntü yükleme, tahmin sonucu, güven skoru ve tüm sınıflar için olasılık dağılımını çubuk grafik olarak sunar. GPU (CUDA) üzerinde tahmin süresi <100 ms'dir.

## 📈 Model Performansı

| Sınıf | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Açık Tekerlekli | 0.994 | 1.000 | 0.997 | 100.0% |
| Hatchback | 0.938 | 0.806 | 0.867 | 80.6% |
| Micro | 0.898 | 0.981 | 0.938 | 98.1% |
| Pick-Up | 0.977 | 0.976 | 0.976 | 97.6% |
| Sedan | 0.900 | 0.953 | 0.926 | 95.3% |
| Station Wagon | 0.935 | 0.929 | 0.932 | 92.9% |
| SUV | 0.900 | 0.953 | 0.926 | 95.3% |
| Van | 1.000 | 0.994 | 0.997 | 99.4% |
| **Makro Ortalama** | **0.943** | **0.949** | **0.945** | **94.61%** |

> Not: Hatchback sınıfı, Sedan ile paylaştığı benzer arka kapı profili ve silüet nedeniyle en düşük performansı göstermektedir (%80.6). Focal Loss ve Mixup'ın birlikte kullanımı bu karışıklığı azaltmış ancak tamamen ortadan kaldıramamıştır.

## 🗂️ Veri Seti

Veri seti iki ana kaynaktan derlenmiştir:

* **Stanford Car Body Type Data:** Marka/model bazlı etiketlenmiş görüntüler gövde tipine göre yeniden sınıflandırılmıştır.
* **Cars Body Type Cropped:** Doğrudan gövde tipi etiketli görüntüler kullanılmıştır.

Ayrıca Station Wagon ve Micro sınıfları için Bing Image Search API ve Roboflow Universe üzerinden ek görüntüler toplanmış, tüm görüntüler manuel olarak incelenerek hatalı etiketler temizlenmiştir.

| Sınıf | Train | Val | Test | Toplam |
|---|---|---|---|---|
| SUV | 1062 | 170 | 170 | 1402 |
| Van | 1000 | 170 | 170 | 1340 |
| Station Wagon | 1001 | 170 | 170 | 1341 |
| Micro | 603 | 54 | 54 | 711 |
| Açık Tekerlekli | 813 | 170 | 170 | 1153 |
| Sedan | 1000 | 170 | 170 | 1340 |
| Hatchback | 1005 | 170 | 170 | 1345 |
| Pick-Up | 1009 | 170 | 170 | 1349 |
| **Toplam** | **7493** | **1244** | **1244** | **9981** |

## 🛠️ Sistem Mimarisi

1. **Veri Ön İşleme Katmanı:** Veri toplama, hash tabanlı tekrar kontrolü, veri artırma pipeline'ı ve ImageNet istatistikleriyle normalizasyon.
2. **Model Katmanı:** EfficientNet-B2 tabanlı, 8 çıktı nöronlu yeniden yapılandırılmış sınıflandırma başlığı (dropout=0.4), Focal Loss ve Mixup augmentation ile desteklenmiş çift aşamalı eğitim.
3. **Servis/Arayüz Katmanı:** Streamlit tabanlı web uygulaması — görüntü yükleme, model çıkarımı (inference) ve sonuç görselleştirme.

## 📋 Kurulum ve Çalıştırma

### Ön Gereksinimler

* Python 3.9+
* PyTorch (CUDA destekli GPU önerilir)
* Streamlit

### Adımlar

1. Proje dizinine gidin:
   ```bash
   cd car-body-classification
   ```

2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Eğitilmiş model ağırlıklarını ilgili dizine yerleştirin (`models/` veya proje yapılandırmasında belirtilen konum).

4. Streamlit uygulamasını başlatın:
   ```bash
   streamlit run app.py
   ```

5. Tarayıcıda açılan arayüzden bir araç görüntüsü yükleyerek gövde tipi tahminini ve güven skorunu görüntüleyin.

## 📚 Kaynak

Bu proje, "Araba Gövde Tipi Sınıflandırma: EfficientNet-B2 Tabanlı Derin Öğrenme Yaklaşımı" başlıklı çalışmaya dayanmaktadır (Z. V. Solmaz, A. Karaaslan, Kocaeli Üniversitesi Bilgisayar Mühendisliği Bölümü).

---

# Car Body Type Classification (EfficientNet-B2)

This project is a deep learning system that automatically classifies vehicle images into **8 different body types** (SUV, Van, Station Wagon, Micro, Convertible/Open-Wheel, Sedan, Hatchback, Pick-Up). The model is built on the **EfficientNet-B2** architecture, fine-tuned using a two-phase transfer learning strategy, and deployed through a **Streamlit**-based web interface for real-time prediction.

## 📊 Screenshots

### Image Upload and Prediction Result
![Streamlit Interface - Prediction](images/streamlit_prediction.png)

### Probability Distribution and Confidence Score
![Probability Distribution](images/probability_distribution.png)

### Supported Classes Panel
![Supported Classes](images/supported_classes.png)

---

## 🚀 Features

* **8-Class Body Type Detection:** Accurately distinguishes SUV, Van, Station Wagon, Micro, Open-Wheel, Sedan, Hatchback, and Pick-Up categories.
* **EfficientNet-B2 Based Architecture:** ~36 MB model size, pretrained ImageNet weights, and 260×260 pixel input resolution deliver higher accuracy at lower computational cost compared to ResNet-50.
* **Two-Phase Training Strategy:** Phase 1 trains only the classification head (backbone frozen); Phase 2 fine-tunes the entire network with adaptive learning-rate decay via ReduceLROnPlateau.
* **Focal Loss + Class Weighting:** Addresses class imbalance and strengthens learning for visually confusable classes such as Hatchback/Micro.
* **Mixup Augmentation:** Linearly blends pairs of training samples and labels (Beta distribution, α=0.3) to produce a smoother decision boundary, particularly improving the Hatchback–Sedan distinction.
* **Extensive Data Augmentation:** RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation, RandomErasing, RandomVerticalFlip, RandomGrayscale, and RandomPerspective improve robustness to real-world conditions.
* **Hash-Based Dataset Deduplication:** Ensures no visual overlap between training, validation, and test sets.
* **Real-Time Web Interface:** A Streamlit app supporting drag-and-drop image upload, prediction with confidence score, and a bar chart of the probability distribution across all classes. Inference time is under 100 ms on GPU (CUDA).

## 📈 Model Performance

| Class | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Open-Wheel | 0.994 | 1.000 | 0.997 | 100.0% |
| Hatchback | 0.938 | 0.806 | 0.867 | 80.6% |
| Micro | 0.898 | 0.981 | 0.938 | 98.1% |
| Pick-Up | 0.977 | 0.976 | 0.976 | 97.6% |
| Sedan | 0.900 | 0.953 | 0.926 | 95.3% |
| Station Wagon | 0.935 | 0.929 | 0.932 | 92.9% |
| SUV | 0.900 | 0.953 | 0.926 | 95.3% |
| Van | 1.000 | 0.994 | 0.997 | 99.4% |
| **Macro Avg.** | **0.943** | **0.949** | **0.945** | **94.61%** |

> Note: The Hatchback class shows the lowest performance (80.6%), largely due to the visual similarity of its rear-door profile and silhouette to the Sedan class. Combining Focal Loss and Mixup reduced this confusion but did not fully eliminate it.

## 🗂️ Dataset

The dataset was compiled from two primary sources:

* **Stanford Car Body Type Data:** Brand/model-labeled images re-classified by body type.
* **Cars Body Type Cropped:** Images directly labeled with body type.

Additional images for the Station Wagon and Micro classes were collected via the Bing Image Search API and Roboflow Universe; all collected images were manually reviewed to remove mislabeled samples.

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| SUV | 1062 | 170 | 170 | 1402 |
| Van | 1000 | 170 | 170 | 1340 |
| Station Wagon | 1001 | 170 | 170 | 1341 |
| Micro | 603 | 54 | 54 | 711 |
| Open-Wheel | 813 | 170 | 170 | 1153 |
| Sedan | 1000 | 170 | 170 | 1340 |
| Hatchback | 1005 | 170 | 170 | 1345 |
| Pick-Up | 1009 | 170 | 170 | 1349 |
| **Total** | **7493** | **1244** | **1244** | **9981** |

## 🛠️ System Architecture

1. **Data Preprocessing Layer:** Data collection, hash-based deduplication, augmentation pipeline, and ImageNet-based normalization.
2. **Model Layer:** EfficientNet-B2 backbone with a rebuilt 8-output classification head (dropout=0.4), trained with Focal Loss and Mixup augmentation across a two-phase training schedule.
3. **Serving/Interface Layer:** A Streamlit-based web application handling image upload, model inference, and result visualization.

## 📋 Installation & Execution

### Prerequisites

* Python 3.9+
* PyTorch (GPU with CUDA recommended)
* Streamlit

### Steps

1. Navigate to the project directory:
   ```bash
   cd car-body-classification
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the trained model weights in the appropriate directory (e.g., `models/`, or as configured in the project).

4. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

5. Upload a vehicle image in the browser interface to view the predicted body type and confidence score.

## 📚 Reference

This project is based on the paper "Car Body Type Classification: A Deep Learning Approach Based on EfficientNet-B2" (Z. V. Solmaz, A. Karaaslan, Department of Computer Engineering, Kocaeli University).
