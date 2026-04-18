import os
import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering, BlipModel

# Make src/ importable when running from repo root
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import Verifier  # noqa: E402

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VQA_MODEL_NAME = "Salesforce/blip-vqa-base"
VERIFIER_CKPT = "outputs/checkpoints/verifier.pt"

st.set_page_config(page_title="VQA + Answer Verification", layout="wide")


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

        # Support wrapped checkpoints or DataParallel
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}

        model.load_state_dict(cleaned)
        model.eval()
        return model, True

    return model, False


processor, vqa_model, feature_model = load_vqa_models()
verifier_model, verifier_loaded = load_verifier()


# -----------------------------
# Functions
# -----------------------------
def generate_answer(image: Image.Image, question: str) -> str:
    inputs = processor(images=image, text=question, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = vqa_model.generate(**inputs, max_new_tokens=5)

    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer.strip()


def _extract_tensor_embedding(emb, fallback_output, which: str) -> torch.Tensor:
    """
    Convert BLIP output into a 2D tensor [batch, hidden_dim].
    Handles cases where emb is already a tensor or a model output object.
    """
    # Preferred case: already a tensor
    if isinstance(emb, torch.Tensor):
        tensor = emb
    # Some model/version combos may return a nested output object
    elif hasattr(emb, "pooler_output") and emb.pooler_output is not None:
        tensor = emb.pooler_output
    elif hasattr(emb, "last_hidden_state") and emb.last_hidden_state is not None:
        tensor = emb.last_hidden_state.mean(dim=1)
    # Fallback to the paired model output object if needed
    elif fallback_output is not None and hasattr(fallback_output, "pooler_output") and fallback_output.pooler_output is not None:
        tensor = fallback_output.pooler_output
    elif fallback_output is not None and hasattr(fallback_output, "last_hidden_state") and fallback_output.last_hidden_state is not None:
        tensor = fallback_output.last_hidden_state.mean(dim=1)
    else:
        raise TypeError(f"Unexpected {which} embedding type: {type(emb)}")

    # Final safety: reduce [batch, seq, hidden] -> [batch, hidden]
    if tensor.dim() == 3:
        tensor = tensor.mean(dim=1)

    return tensor


def get_embeddings(image: Image.Image, question: str, answer: str):
    """
    Returns:
        img_emb: [1, 512]
        txt_emb: [1, 512]
    """
    text_input = question + " " + answer

    proc = processor(
        images=image,
        text=text_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    proc = {k: v.to(DEVICE) for k, v in proc.items()}

    with torch.no_grad():
        out = feature_model(**proc, return_dict=True)

        # BLIP forward output may expose embeddings directly, while nested outputs
        # are BaseModelOutputWithPooling objects.
        img_emb = _extract_tensor_embedding(
            getattr(out, "image_embeds", None),
            getattr(out, "vision_model_output", None),
            "image",
        )
        txt_emb = _extract_tensor_embedding(
            getattr(out, "text_embeds", None),
            getattr(out, "text_model_output", None),
            "text",
        )

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
                "Verifier checkpoint not found. Run `src/train_verifier.py` first "
                "to enable verification."
            )
else:
    st.info("Please upload an image to start.")
