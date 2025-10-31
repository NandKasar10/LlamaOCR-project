# app.py
import streamlit as st
import requests
import base64
import cv2
import numpy as np
import pytesseract
from PIL import Image
from io import BytesIO
from pathlib import Path
import json

# ----------- Configuration -----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llava"

st.set_page_config(page_title="LlamaOCR", layout="centered")

# ---------- Image preprocessing ----------
def preprocess_image_bytes(image_bytes: bytes) -> bytes:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)

    # Denoising
    den = cv2.fastNlMeansDenoising(cl, None, h=10)

    # Thresholding
    _, th = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Upscale slightly for better OCR
    upscale = cv2.resize(th, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Encode to PNG bytes
    is_success, buffer = cv2.imencode(".png", upscale)
    return buffer.tobytes()

# ---------- Tesseract OCR ----------
def tesseract_candidates(img_bytes: bytes, psm_list=(6, 7, 11)):
    pil = Image.open(BytesIO(img_bytes)).convert("L")
    results = []
    confidences = []
    for psm in psm_list:
        cfg = f'--oem 1 --psm {psm}'
        raw = pytesseract.image_to_string(pil, config=cfg).strip()
        data = pytesseract.image_to_data(pil, config=cfg, output_type=pytesseract.Output.DICT)
        
        # compute mean confidence (ignore invalid)
        confs = []
        for c in data.get('conf', []):
            try:
                val = float(c)
                if val >= 0:
                    confs.append(int(val))
            except (ValueError, TypeError):
                continue

        mean_conf = (sum(confs) / len(confs)) if confs else 0
        if raw:
            results.append(raw)
            confidences.append(mean_conf)
    
    # Deduplicate
    seen = set()
    uniq = []
    for r, c in zip(results, confidences):
        if r not in seen:
            seen.add(r)
            uniq.append((r, c))
    return uniq

# ---------- LLaVA Call ----------
def call_llava(image_bytes: bytes, timeout=60):
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a highly accurate OCR assistant that extracts text from any image, including handwritten or printed text."},
            {"role": "user", "content": "Extract and transcribe all readable text from the uploaded image clearly. Maintain correct spelling and structure."}
        ],
        "images": [image_b64]
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    return r

def extract_llava_text(resp):
    try:
        j = resp.json()
    except Exception:
        return ""
    if isinstance(j, dict):
        msg = j.get("message") or {}
        content = msg.get("content") if isinstance(msg, dict) else ""
        if content:
            return content.strip()
        if "choices" in j:
            for c in j["choices"]:
                if isinstance(c, dict) and "delta" in c:
                    cont = c.get("delta", {}).get("content")
                    if cont:
                        return cont.strip()
    return json.dumps(j)[:1000]

# ---------- Merge ----------
def merge_results(tess_list, llava_text):
    best_tess = tess_list[0][0] if tess_list else ""
    best_tess_conf = tess_list[0][1] if tess_list else 0
    ll = llava_text.strip()
    candidates = []

    if ll and best_tess:
        if normalize_text(ll) == normalize_text(best_tess):
            candidates.append((ll, "High", "LLaVA + Tesseract agree"))
        else:
            candidates.append((ll, "Medium", "LLaVA (preferred)"))
            candidates.append((best_tess, "Medium", f"Tesseract (conf {int(best_tess_conf)})"))
    elif ll:
        candidates.append((ll, "Medium", "LLaVA"))
    elif best_tess:
        conf_label = "High" if best_tess_conf >= 80 else ("Medium" if best_tess_conf >= 40 else "Low")
        candidates.append((best_tess, conf_label, f"Tesseract (conf {int(best_tess_conf)})"))

    for txt, conf in tess_list[1:3]:
        if normalize_text(txt) not in [normalize_text(c[0]) for c in candidates]:
            conf_label = "High" if conf >= 80 else ("Medium" if conf >= 40 else "Low")
            candidates.append((txt, conf_label, f"Tesseract (conf {int(conf)})"))
    if not candidates:
        candidates.append(("", "Low", "No transcription"))
    return candidates

def normalize_text(s):
    return "".join(s.lower().split())

# ---------- UI ----------
st.header("📄 LlamaOCR — AI-Powered OCR Assistant")

uploaded = st.file_uploader("Upload any image (text, handwriting, document)", type=["png", "jpg", "jpeg"])
if uploaded:
    st.subheader("Input Preview")
    st.image(uploaded, use_column_width=True)

    if st.button("Run OCR"):
        with st.spinner("🔧 Preprocessing image..."):
            try:
                pre_bytes = preprocess_image_bytes(uploaded.read())
            except Exception as e:
                st.error("Preprocessing failed: " + str(e))
                st.stop()

        st.subheader("Processed Image for OCR")
        st.image(pre_bytes, use_column_width=True)

        with st.spinner("🔍 Running Tesseract OCR..."):
            tess_list = tesseract_candidates(pre_bytes)
        st.info(f"Tesseract found {len(tess_list)} possible output(s).")

        with st.spinner("🤖 Querying LLaVA model..."):
            try:
                resp = call_llava(pre_bytes)
            except Exception as e:
                st.error("Error contacting LLaVA: " + str(e))
                resp = None

        if resp is None:
            st.error("No response from LLaVA.")
            st.stop()

        st.write("HTTP:", resp.status_code)
        llava_text = extract_llava_text(resp)
        candidates = merge_results(tess_list, llava_text)

        best, best_conf, best_prov = candidates[0]
        alts = candidates[1:3]
        provenance = f"{best_prov}; LLaVA: {bool(llava_text)}; Tesseract: {len(tess_list)}"

        md = "### Transcription Results\n\n"
        md += f"**Best Guess:**\n\n```\n{best}\n```\n\n"
        if alts:
            md += "**Alternatives:**\n"
            for a, c, p in alts:
                md += f"- `{a}` — *{c} confidence* ({p})\n"
        md += f"\n**Source:** {provenance}\n"
        st.markdown(md)

        edited = st.text_area("📝 Edit or Confirm Transcription", value=best, height=100)
        if st.button("✅ Save Transcription"):
            Path("transcriptions").mkdir(exist_ok=True)
            fname = Path("transcriptions") / f"transcription_{np.random.randint(1e9)}.txt"
            fname.write_text(edited, encoding="utf-8")
            st.success(f"Saved successfully to {fname}")
