import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import config

st.set_page_config(
    page_title="CarVision — Araba Gövde Sınıflandırma",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    box-sizing: border-box;
}

.stApp {
    background: #0b0f1a;
    color: #e8eaf0;
}

/* ── HEADER ── */
.cv-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2rem 2.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 2rem;
}
.cv-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    color: #fff;
    letter-spacing: -1px;
}
.cv-logo span { color: #4f8ef7; }
.cv-meta {
    color: rgba(255,255,255,0.35);
    font-size: 0.82rem;
    text-align: right;
    line-height: 1.6;
}
.cv-chips {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
    justify-content: flex-end;
}
.cv-chip {
    background: rgba(79,142,247,0.12);
    border: 1px solid rgba(79,142,247,0.3);
    color: #7eb3ff;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}
.cv-chip.green {
    background: rgba(52,211,153,0.1);
    border-color: rgba(52,211,153,0.3);
    color: #34d399;
}

/* ── CARDS ── */
.cv-card {
    background: #141929;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.cv-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: rgba(255,255,255,0.4);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── UPLOAD AREA ── */
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(79,142,247,0.3) !important;
    border-radius: 12px !important;
    background: rgba(79,142,247,0.04) !important;
    transition: all 0.2s !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(79,142,247,0.6) !important;
    background: rgba(79,142,247,0.08) !important;
}
div[data-testid="stFileUploader"] section {
    background: transparent !important;
}

/* ── RESULT CARD ── */
.cv-result {
    background: linear-gradient(135deg, #141929 0%, #1a2240 100%);
    border: 1px solid rgba(79,142,247,0.25);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.cv-result::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4f8ef7, #a78bfa, #4f8ef7);
}
.cv-result-label {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.3);
    margin-bottom: 0.5rem;
}
.cv-result-class {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #fff;
    line-height: 1.1;
    margin-bottom: 0.8rem;
}
.cv-result-conf {
    display: inline-block;
    background: rgba(79,142,247,0.15);
    border: 1px solid rgba(79,142,247,0.4);
    color: #7eb3ff;
    padding: 0.4rem 1.2rem;
    border-radius: 20px;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
.cv-result-sub {
    color: rgba(255,255,255,0.3);
    font-size: 0.8rem;
}

/* ── METRIC GRID ── */
.cv-metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}
.cv-metric {
    background: #141929;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.cv-metric-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: rgba(255,255,255,0.3);
    margin-bottom: 0.4rem;
}
.cv-metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e8eaf0;
}
.cv-metric-value.blue { color: #4f8ef7; }
.cv-metric-value.green { color: #34d399; }
.cv-metric-value.purple { color: #a78bfa; }

/* ── PERFORMANCE TABLE ── */
.cv-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
}
.cv-table th {
    padding: 0.75rem 1rem;
    text-align: left;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: rgba(255,255,255,0.35);
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.cv-table td {
    padding: 0.7rem 1rem;
    color: #c8cdd8;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.cv-table tr:last-child td { border-bottom: none; }
.cv-table tr:hover td { background: rgba(79,142,247,0.05); }
.cv-table .cls-name {
    font-weight: 600;
    color: #e8eaf0;
}
.f1-high { color: #34d399; font-weight: 700; }
.f1-mid  { color: #fbbf24; font-weight: 700; }
.f1-low  { color: #f87171; font-weight: 700; }

/* ── F1 BAR ── */
.f1-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.f1-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 3px;
    overflow: hidden;
}
.f1-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #4f8ef7, #a78bfa);
}

/* ── CLASS GRID ── */
.cv-class-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
}
.cv-class-item {
    background: #141929;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.cv-class-item:hover {
    border-color: rgba(79,142,247,0.35);
}
.cv-class-emoji { font-size: 2rem; margin-bottom: 0.4rem; }
.cv-class-code {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #4f8ef7;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.cv-class-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: #e8eaf0;
}

/* ── EMPTY STATE ── */
.cv-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 1rem;
    color: rgba(255,255,255,0.2);
    border: 2px dashed rgba(255,255,255,0.06);
    border-radius: 12px;
    text-align: center;
}
.cv-empty-icon { font-size: 2.5rem; margin-bottom: 0.6rem; opacity: 0.5; }

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #3b7de8, #4f8ef7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 24px rgba(79,142,247,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4f8ef7, #6ea8ff) !important;
    box-shadow: 0 6px 30px rgba(79,142,247,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    background: transparent;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: rgba(255,255,255,0.4);
    border-radius: 8px 8px 0 0;
    padding: 0.6rem 1.4rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.9rem;
    border: 1px solid transparent;
    border-bottom: none;
}
.stTabs [aria-selected="true"] {
    background: #141929 !important;
    color: #7eb3ff !important;
    border-color: rgba(79,142,247,0.25) !important;
    border-bottom: none !important;
}

/* ── IMAGE ── */
.stImage img {
    border-radius: 12px;
}

/* ── FOOTER ── */
.cv-footer {
    text-align: center;
    color: rgba(255,255,255,0.2);
    font-size: 0.75rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 3rem;
}

hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MODEL YÜKLEME
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    model_path = PROJECT_ROOT / "models" / "best_model.pth"
    if not model_path.exists():
        return None, None
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    model_name = checkpoint.get("model_name", config.MODEL_NAME)
    class_names = checkpoint["class_names"]
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.DEVICE)
    model.eval()
    return model, class_names


def get_transform():
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


@torch.no_grad()
def predict(model, image, class_names):
    start = time.time()
    tensor = get_transform()(image).unsqueeze(0).to(config.DEVICE)
    probs = F.softmax(model(tensor), dim=1)[0]
    confidence, pred_idx = probs.max(0)
    ms = (time.time() - start) * 1000
    pred_cls = class_names[pred_idx.item()]
    return {
        "predicted_class": pred_cls,
        "display_name": config.CLASS_DISPLAY_NAMES.get(pred_cls, pred_cls),
        "confidence": confidence.item(),
        "probabilities": {class_names[i]: probs[i].item() for i in range(len(class_names))},
        "class_code": config.CLASS_TO_CODE.get(pred_cls, 0),
        "inference_time_ms": ms,
    }


# ══════════════════════════════════════════════════════════════════
# RESULTS PARSE
# ══════════════════════════════════════════════════════════════════
def find_latest_results_dir():
    candidates = sorted(
        [d for d in PROJECT_ROOT.iterdir() if d.is_dir() and d.name.startswith("results")],
        key=lambda d: d.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def parse_report(report_path):
    if not report_path.exists():
        return None
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    data = {"test_acc": None, "test_loss": None, "per_class": [], "macro": {}, "weighted": {}}
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("Test Loss:"):
            try: data["test_loss"] = float(line.split(":")[1].strip())
            except: pass
        elif line.startswith("Test Accuracy:"):
            try: data["test_acc"] = float(line.split(":")[1].strip().replace("%", ""))
            except: pass
        else:
            parts = line.split()
            if len(parts) >= 5:
                try:
                    name = " ".join(parts[:-4])
                    prec, rec, f1, sup = parts[-4:]
                    row = {"name": name, "precision": float(prec),
                           "recall": float(rec), "f1": float(f1), "support": int(sup)}
                    if "macro avg" in name:
                        data["macro"] = row
                    elif "weighted avg" in name:
                        data["weighted"] = row
                    elif name not in ("accuracy",):
                        data["per_class"].append(row)
                except: pass
    return data


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
model, class_names = load_model()
device_str = "GPU (CUDA)" if config.DEVICE.type == "cuda" else "CPU"
model_ok = model is not None

st.markdown(f"""
<div class="cv-header">
    <div>
        <div class="cv-logo">Car<span>Vision</span></div>
        <div style="color:rgba(255,255,255,0.3); font-size:0.8rem; margin-top:0.2rem;">
            Araba Gövde Tipi Sınıflandırma Sistemi
        </div>
    </div>
    <div class="cv-meta">
        Kocaeli Üniversitesi · Bilgisayar Mühendisliği · Yazılım Lab. II · 2026
        <div class="cv-chips">
            <span class="cv-chip">EfficientNet-B2</span>
            <span class="cv-chip">8 Sınıf</span>
            <span class="cv-chip">⚡ {device_str}</span>
            <span class="cv-chip {'green' if model_ok else ''}">
                {'✓ Model Hazır' if model_ok else '✗ Model Yok'}
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_ok:
    st.error("⚠️ Model dosyası bulunamadı: `models/best_model.pth` — önce `python src/train.py` çalıştırın.")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# SEKMELER
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍  Tahmin", "📊  Model Performansı", "📋  Sınıflar"])


# ──────────────────────────────────────────────────────────────────
# TAB 1 — TAHMİN
# ──────────────────────────────────────────────────────────────────
with tab1:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="cv-card"><div class="cv-card-title">📁 Görsel Yükle</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Araba görseli seçin",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed"
        )
        if uploaded:
            try:
                image = Image.open(uploaded).convert("RGB")
            except:
                st.error("Görsel açılamadı.")
                st.stop()
            st.image(image, use_container_width=True)
            st.markdown(
                f'<div style="text-align:center; color:rgba(255,255,255,0.3); '
                f'font-size:0.8rem; margin-top:0.5rem;">'
                f'{uploaded.name} &nbsp;·&nbsp; {image.size[0]}×{image.size[1]} px</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown("""
            <div class="cv-empty" style="height:260px;">
                <div class="cv-empty-icon">🖼️</div>
                <div style="font-size:0.9rem;">Görsel bekleniyor</div>
                <div style="font-size:0.75rem; margin-top:0.3rem; opacity:0.6;">JPG · PNG · WEBP · BMP</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="cv-card"><div class="cv-card-title">🔍 Tahmin Sonucu</div>', unsafe_allow_html=True)

        if uploaded:
            if st.button("🔍  Sınıflandır", type="primary", use_container_width=True):
                with st.spinner("Analiz ediliyor..."):
                    result = predict(model, image, class_names)

                emoji_map = {
                    "SUV": "🚙", "VAN": "🚐", "STATION_WAGON": "🚗",
                    "MICRO": "🚘", "ACIK_TEKERLEKLI": "🏎️",
                    "SEDAN": "🚖", "HATCHBACK": "🚕", "PICK_UP": "🛻",
                }
                emoji = emoji_map.get(result["predicted_class"], "🚗")

                st.markdown(f"""
                <div class="cv-result">
                    <div style="font-size:3rem; margin-bottom:0.4rem;">{emoji}</div>
                    <div class="cv-result-label">Tahmin Edilen Sınıf</div>
                    <div class="cv-result-class">{result['display_name']}</div>
                    <div class="cv-result-conf">%{result['confidence']*100:.1f} güven</div>
                    <div class="cv-result-sub">
                        Sınıf Kodu: {result['class_code']} &nbsp;·&nbsp; ⚡ {result['inference_time_ms']:.0f} ms
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Olasılık bar chart
                st.markdown('<div class="cv-card-title" style="margin-bottom:0.8rem;">📊 Olasılık Dağılımı</div>', unsafe_allow_html=True)
                probs = result["probabilities"]
                sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                labels = [config.CLASS_DISPLAY_NAMES.get(k, k) for k, _ in sorted_p]
                values = [v * 100 for _, v in sorted_p]
                pred_disp = result["display_name"]
                colors = ["#4f8ef7" if config.CLASS_DISPLAY_NAMES.get(k, k) == pred_disp
                          else "rgba(79,142,247,0.2)" for k, _ in sorted_p]

                fig = go.Figure(go.Bar(
                    x=values, y=labels, orientation="h",
                    marker=dict(color=colors),
                    text=[f"  {v:.1f}%" for v in values],
                    textposition="outside",
                    textfont=dict(color="rgba(255,255,255,0.6)", size=11),
                ))
                fig.update_layout(
                    height=310, margin=dict(l=5, r=55, t=5, b=20),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(range=[0, 125], tickfont=dict(color="rgba(255,255,255,0.3)", size=10),
                               gridcolor="rgba(255,255,255,0.05)", title_font=dict(color="rgba(255,255,255,0.3)"),
                               title="Olasılık (%)"),
                    yaxis=dict(tickfont=dict(color="rgba(255,255,255,0.7)", size=11)),
                    bargap=0.3, font=dict(color="white"),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Mini metrik satırı
                st.markdown(f"""
                <div class="cv-metric-grid">
                    <div class="cv-metric">
                        <div class="cv-metric-label">Sınıf</div>
                        <div class="cv-metric-value" style="font-size:1rem;">{result['display_name']}</div>
                    </div>
                    <div class="cv-metric">
                        <div class="cv-metric-label">Güven</div>
                        <div class="cv-metric-value blue">%{result['confidence']*100:.1f}</div>
                    </div>
                    <div class="cv-metric">
                        <div class="cv-metric-label">Kod</div>
                        <div class="cv-metric-value purple">{result['class_code']}</div>
                    </div>
                    <div class="cv-metric">
                        <div class="cv-metric-label">Süre</div>
                        <div class="cv-metric-value green">{result['inference_time_ms']:.0f} ms</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="cv-empty" style="height:200px;">
                    <div class="cv-empty-icon">🔍</div>
                    <div>Butona basarak tahmin yapın</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="cv-empty" style="height:260px;">
                <div class="cv-empty-icon">⬅️</div>
                <div>Önce sol taraftan görsel yükleyin</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# TAB 2 — MODEL PERFORMANSI
# ──────────────────────────────────────────────────────────────────
with tab2:
    results_dir = find_latest_results_dir()

    if results_dir is None:
        st.warning("⚠️ Sonuç klasörü bulunamadı. Önce `python src/train.py` çalıştırın.")
    else:
        report = parse_report(results_dir / "classification_report.txt")

        if report:
            macro = report.get("macro", {})
            weighted = report.get("weighted", {})

            # ── Genel metrikler ──
            st.markdown('<div class="cv-card">', unsafe_allow_html=True)
            st.markdown('<div class="cv-card-title">📈 Genel Performans</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="cv-metric-grid">
                <div class="cv-metric">
                    <div class="cv-metric-label">Test Accuracy</div>
                    <div class="cv-metric-value blue">%{report.get('test_acc', 0):.2f}</div>
                </div>
                <div class="cv-metric">
                    <div class="cv-metric-label">Macro F1-Score</div>
                    <div class="cv-metric-value green">{macro.get('f1', 0):.4f}</div>
                </div>
                <div class="cv-metric">
                    <div class="cv-metric-label">Macro Precision</div>
                    <div class="cv-metric-value">{macro.get('precision', 0):.4f}</div>
                </div>
                <div class="cv-metric">
                    <div class="cv-metric-label">Macro Recall</div>
                    <div class="cv-metric-value">{macro.get('recall', 0):.4f}</div>
                </div>
            </div>
            <div class="cv-metric-grid" style="margin-top:0;">
                <div class="cv-metric">
                    <div class="cv-metric-label">Weighted F1</div>
                    <div class="cv-metric-value purple">{weighted.get('f1', 0):.4f}</div>
                </div>
                <div class="cv-metric">
                    <div class="cv-metric-label">Weighted Precision</div>
                    <div class="cv-metric-value">{weighted.get('precision', 0):.4f}</div>
                </div>
                <div class="cv-metric">
                    <div class="cv-metric-label">Weighted Recall</div>
                    <div class="cv-metric-value">{weighted.get('recall', 0):.4f}</div>
                </div>
                <div class="cv-metric">
                    <div class="cv-metric-label">Test Loss</div>
                    <div class="cv-metric-value">{report.get('test_loss', 0):.4f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Sınıf bazında tablo ──
            st.markdown('<div class="cv-card">', unsafe_allow_html=True)
            st.markdown('<div class="cv-card-title">🎯 Sınıf Bazında Metrikler</div>', unsafe_allow_html=True)

            rows_html = ""
            for row in report["per_class"]:
                f1 = row["f1"]
                f1_cls = "f1-high" if f1 >= 0.93 else "f1-mid" if f1 >= 0.85 else "f1-low"
                bar_pct = int(f1 * 100)
                bar_color = "#34d399" if f1 >= 0.93 else "#fbbf24" if f1 >= 0.85 else "#f87171"
                disp = config.CLASS_DISPLAY_NAMES.get(row["name"], row["name"])
                rows_html += f"""
                <tr>
                    <td class="cls-name">{disp}</td>
                    <td>{row['precision']:.4f}</td>
                    <td>{row['recall']:.4f}</td>
                    <td>
                        <div class="f1-bar-wrap">
                            <div class="f1-bar-bg">
                                <div class="f1-bar-fill" style="width:{bar_pct}%; background:{bar_color};"></div>
                            </div>
                            <span class="{f1_cls}">{row['f1']:.4f}</span>
                        </div>
                    </td>
                    <td style="color:rgba(255,255,255,0.4);">{row['support']}</td>
                </tr>"""

            st.markdown(f"""
            <table class="cv-table">
                <thead><tr>
                    <th>Sınıf</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Eğitim Grafikleri ──
        st.markdown('<div class="cv-card">', unsafe_allow_html=True)
        st.markdown('<div class="cv-card-title">📉 Eğitim Grafikleri</div>', unsafe_allow_html=True)

        g1, g2 = st.tabs(["Loss", "Accuracy"])
        with g1:
            lp = results_dir / "loss_plot.png"
            if lp.exists():
                st.image(str(lp), use_container_width=True)
                st.caption("Training & Validation Loss — epoch bazında kayıp değişimi. İki eğri arasındaki farkın küçük olması overfitting olmadığını gösterir.")
            else:
                st.info("Loss grafiği henüz mevcut değil.")
        with g2:
            ap = results_dir / "accuracy_plot.png"
            if ap.exists():
                st.image(str(ap), use_container_width=True)
                st.caption("Training & Validation Accuracy — epoch bazında doğruluk değişimi.")
            else:
                st.info("Accuracy grafiği henüz mevcut değil.")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Confusion Matrix ──
        st.markdown('<div class="cv-card">', unsafe_allow_html=True)
        st.markdown('<div class="cv-card-title">🗂️ Confusion Matrix</div>', unsafe_allow_html=True)

        cm_norm = results_dir / "confusion_matrix_normalized.png"
        cm_raw  = results_dir / "confusion_matrix_raw.png"

        cm1, cm2 = st.columns(2)
        with cm1:
            if cm_norm.exists():
                st.image(str(cm_norm), use_container_width=True)
                st.caption("Normalize Edilmiş — satırlar gerçek, sütunlar tahmin edilen sınıfları gösterir. Diyagonal değerler 1.00'a yakın olmalı.")
            else:
                st.info("Normalize confusion matrix mevcut değil.")
        with cm2:
            if cm_raw.exists():
                st.image(str(cm_raw), use_container_width=True)
                st.caption("Ham Sayılar — her hücredeki görüntü adedi.")
            else:
                st.info("Ham confusion matrix mevcut değil.")

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Sınıf Detay Grafikleri ──
        pca = results_dir / "per_class_accuracy.png"
        pcm = results_dir / "per_class_metrics.png"
        if pca.exists() or pcm.exists():
            st.markdown('<div class="cv-card">', unsafe_allow_html=True)
            st.markdown('<div class="cv-card-title">📌 Sınıf Detay Grafikleri</div>', unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            with d1:
                if pca.exists():
                    st.image(str(pca), use_container_width=True)
                    st.caption("Sınıf bazında doğruluk oranı")
            with d2:
                if pcm.exists():
                    st.image(str(pcm), use_container_width=True)
                    st.caption("Sınıf bazında Precision / Recall / F1")
            st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# TAB 3 — SINIFLAR
# ──────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="cv-card">', unsafe_allow_html=True)
    st.markdown('<div class="cv-card-title">📋 Desteklenen 8 Sınıf</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:rgba(255,255,255,0.35); font-size:0.85rem; margin-bottom:1.2rem;">'
        'Model aşağıdaki 8 araba gövde tipini tanıyabilir. '
        'Sunum sırasında test scripti sınıf kodlarını kullanır.</p>',
        unsafe_allow_html=True
    )

    classes = [
        ("1", "SUV", "🚙"),
        ("2", "Van", "🚐"),
        ("3", "Station Wagon", "🚗"),
        ("4", "Micro", "🚘"),
        ("5", "Açık Tekerlekli (F1)", "🏎️"),
        ("6", "Sedan", "🚖"),
        ("7", "Hatchback", "🚕"),
        ("8", "Pick-Up", "🛻"),
    ]

    st.markdown('<div class="cv-class-grid">', unsafe_allow_html=True)
    for kod, isim, emoji in classes:
        st.markdown(f"""
        <div class="cv-class-item">
            <div class="cv-class-emoji">{emoji}</div>
            <div class="cv-class-code">KOD {kod}</div>
            <div class="cv-class-name">{isim}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="cv-footer">
    CarVision &nbsp;·&nbsp; Kocaeli Üniversitesi Bilgisayar Mühendisliği
    &nbsp;·&nbsp; Yazılım Laboratuvarı II &nbsp;·&nbsp; 2026
</div>
""", unsafe_allow_html=True)