import os, numpy as np, torch, torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import google.generativeai as genai

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
EMOJI   = {"cardboard":"📦","glass":"🧪","metal":"⚙️","paper":"📄","plastic":"♻️","trash":"🗑️"}
IMG_SIZE   = 300
MODEL_PATH = "efficientnet_b3_waste.pth"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="♻️ SmartSort AI", page_icon="♻️", layout="wide")
st.markdown(f"**Device:** `{device}`  **Model:** EfficientNet-B3  **Classes:** 6")

@st.cache_resource
def load_model():
    m = models.efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(p=0.4,inplace=True),
                                  nn.Linear(m.classifier[1].in_features,len(CLASSES)))
    if os.path.exists(MODEL_PATH):
        m.load_state_dict(torch.load(MODEL_PATH,map_location=device,weights_only=True))
        st.sidebar.success("✅ Trained weights loaded")
    else:
        st.sidebar.warning("⚠️ No weights — run notebook first")
    return m.eval().to(device)

tfm = T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)),T.ToTensor(),
                 T.Normalize([.485,.456,.406],[.229,.224,.225])])

def predict(model,img):
    t = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad(): p = torch.softmax(model(t),1).squeeze().cpu().numpy()
    i = int(np.argmax(p))
    return CLASSES[i],float(p[i]),p,i

def gradcam(model,img,idx):
    cam = GradCAM(model=model,target_layers=[model.features[-1]])
    t   = tfm(img).unsqueeze(0).to(device)
    rgb = np.array(img.resize((IMG_SIZE,IMG_SIZE)))/255.0
    g   = cam(input_tensor=t,targets=[ClassifierOutputTarget(idx)])
    return show_cam_on_image(rgb.astype(np.float32),g[0],use_rgb=True)

def ask_gemini(cls,conf,key):
    if not key.strip(): return "_Enter Gemini API key in sidebar._"
    try:
        genai.configure(api_key=key)
        r = genai.GenerativeModel("gemini-2.5-flash-lite").generate_content(
            f"Waste item identified as **{cls}** ({conf*100:.1f}% confidence). Provide: 1) ♻️ Proper disposal method 2) 🌍 Environmental impact if wrong 3) 💡 Creative reuse idea 4) 📊 Recyclability score /10. Be concise.")
        return r.text
    except Exception as e: return f"Error: {e}"

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    key   = st.text_input("Gemini API Key",type="password")
    gcam  = st.toggle("GradCAM Heatmap",value=True)
    bar   = st.toggle("Probability Chart",value=True)
    ai    = st.toggle("Gemini Advice",value=True)
    st.divider()
    st.markdown(f"**Device:** `{device}`  **Model:** EfficientNet-B3  **Classes:** 12")

st.markdown("<h1>♻️ SmartSort AI</h1>",unsafe_allow_html=True)
st.markdown("**EfficientNet-B3 · GradCAM · Gemini AI**")
st.divider()
model = load_model()
up    = st.file_uploader("Upload waste image",type=["jpg","jpeg","png","webp"])

if up:
    img = Image.open(up).convert("RGB")
    c1,c2 = st.columns(2,gap="large")
    with c1: st.image(img,caption="Uploaded",use_column_width=True)
    with st.spinner("Classifying..."):
        pc,cf,ap,ci = predict(model,img)
    with c2:
        st.markdown(f"### {EMOJI.get(pc,"♻️")} Prediction")
        st.markdown(f"<div class='card'><div class='cval'>{pc.upper()}</div>"
                    f"<div class='clbl'>Confidence: {cf*100:.1f}%</div></div>",
                    unsafe_allow_html=True)
        top3 = np.argsort(ap)[::-1][:3]
        cols = st.columns(3)
        for i,idx in enumerate(top3):
            with cols[i]:
                st.markdown(f"<div class='card'><div class='cval' style='font-size:1.1rem'>"
                            f"{ap[idx]*100:.1f}%</div><div class='clbl'>{CLASSES[idx]}</div></div>",
                            unsafe_allow_html=True)
    st.divider()
    if gcam:
        st.markdown("### 🔥 GradCAM — *Why did the model predict this?*")
        with st.spinner("Generating heatmap..."):
            try:
                ov = gradcam(model,img,ci)
                g1,g2 = st.columns(2)
                with g1: st.image(img.resize((IMG_SIZE,IMG_SIZE)),caption="Original",use_column_width=True)
                with g2: st.image(ov,caption="GradCAM (red = most influential)",use_column_width=True)
            except Exception as e: st.warning(f"GradCAM error: {e}")
    if bar:
        st.markdown("### 📊 Probability Distribution")
        fig,ax = plt.subplots(figsize=(10,3))
        fig.patch.set_facecolor("#161b22"); ax.set_facecolor("#161b22")
        colors = ["#39d353" if c==pc else "#30363d" for c in CLASSES]
        ax.barh(CLASSES,ap*100,color=colors,height=0.6)
        ax.set_xlabel("Probability (%)",color="#8b949e")
        ax.tick_params(colors="#8b949e"); ax.spines[:].set_color("#30363d")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    if ai:
        st.markdown("### 🤖 Gemini AI Disposal Advice")
        with st.spinner("Asking Gemini..."):
            adv = ask_gemini(pc,cf,key)
        st.markdown(f"<div class='gbox'>{adv}</div>",unsafe_allow_html=True)
else:
    st.markdown("### 🗂️ 6 Waste Categories")
    cols = st.columns(6)
    for i,cls in enumerate(CLASSES):
        with cols[i%6]:
            st.markdown(f"<div class='card'><div style='font-size:1.5rem'>{EMOJI.get(cls,'♻️')}</div>"
                        f"<div class='clbl'>{cls}</div></div><br>",unsafe_allow_html=True)
    st.info("👆 Upload an image above to classify it!")