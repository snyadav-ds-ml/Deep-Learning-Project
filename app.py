import os
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipModel

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VQA_MODEL_NAME = "Salesforce/blip-vqa-base"
VERIFIER_CKPT = "outputs/checkpoints/verifier.pt"

st.set_page_config(page_title="VQA + Answer Verification", layout="wide")


# -----------------------------
# Verifier Model
# -----------------------------
class Verifier(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, img_emb, txt_emb):
        x = torch.cat([img_emb, txt_emb], dim=1)
        return self.net(x)


# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_vqa_models():
    processor = BlipProcessor.from_pretrained(VQA_MODEL_NAME)
    vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL_NAME).to(DEVICE)
    feature_model = BlipModel.from_pretrained(VQA_MODEL_NAME).to(DEVICE)

    vqa_model.eval()
    feature_model.eval()
    return processor, vqa_model, feature_model


@st.cache_resource
def load_verifier():
    model = Verifier(emb_dim=512).to(DEVICE)
    if os.path.exists(VERIFIER_CKPT):
        state = torch.load(VERIFIER_CKPT, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        return model, True
    return model, False


processor, vqa_model, feature_model = load_vqa_models()
verifier_model, verifier_loaded = load_verifier()


# -----------------------------
# Functions
# -----------------------------
def generate_answer(image: Image.Image, question: str) -> str:
    inputs = processor(images=image, text=question, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = vqa_model.generate(**inputs, max_new_tokens=5)

    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer.strip()


def get_embeddings(image: Image.Image, question: str, answer: str):
    """
    Returns:
        img_emb: [1, emb_dim]
        txt_emb: [1, emb_dim]
    """
    text_input = question + " " + answer

    proc = processor(
        images=image,
        text=text_input,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    proc = {k: v.to(DEVICE) for k, v in proc.items()}

    with torch.no_grad():
        out = feature_model(**proc, return_dict=True)

        img_emb = out.image_embeds
        txt_emb = out.text_embeds

        if hasattr(img_emb, "last_hidden_state"):
            img_emb = img_emb.last_hidden_state
        if hasattr(txt_emb, "last_hidden_state"):
            txt_emb = txt_emb.last_hidden_state

        if img_emb.dim() == 3:
            img_emb = img_emb.mean(dim=1)
        if txt_emb.dim() == 3:
            txt_emb = txt_emb.mean(dim=1)

    return img_emb, txt_emb


def verify_answer(image: Image.Image, question: str, answer: str):
    """
    Returns:
        label: "Correct-like" or "Incorrect-like"
        score: probability in [0, 1]
    """
    if not verifier_loaded:
        return "Verifier not loaded", None

    img_emb, txt_emb = get_embeddings(image, question, answer)

    with torch.no_grad():
        logit = verifier_model(img_emb, txt_emb).squeeze(-1)
        prob = torch.sigmoid(logit).item()

    label = "Correct-like" if prob >= 0.5 else "Incorrect-like"
    return label, prob


# -----------------------------
# UI
# -----------------------------
st.title("Multimodal VQA with Answer Verification")
st.write(
    "Upload an image, ask a question, generate an answer, and check whether the answer "
    "is likely correct using the trained verification model."
)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

with col2:
    question = st.text_input("Enter your question", value="What is in the image?")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Answer"):
        with st.spinner("Generating answer..."):
            answer = generate_answer(image, question)

        st.subheader("Generated Answer")
        st.success(answer if answer else "[No answer generated]")

        st.subheader("Verification")
        if verifier_loaded:
            with st.spinner("Checking answer consistency..."):
                label, score = verify_answer(image, question, answer)

            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence score:** {score:.4f}")

            if score >= 0.5:
                st.progress(min(score, 1.0))
            else:
                st.progress(min(1.0 - score, 1.0))
        else:
            st.warning(
                "Verifier checkpoint not found. Run `train_verifier.py` first "
                "to enable verification."
            )

else:
    st.info("Please upload an image to start.")