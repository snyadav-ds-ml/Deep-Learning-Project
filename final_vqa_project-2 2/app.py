import os
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipModel, BlipProcessor

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models import MultimodalVerifier
from utils import encode_text, load_json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLIP_NAME = "Salesforce/blip-vqa-base"

st.set_page_config(page_title="Multimodal VQA + Verifier", layout="wide")


@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained(BLIP_NAME)
    vqa_model = BlipForQuestionAnswering.from_pretrained(BLIP_NAME).to(DEVICE)
    vision_model = BlipModel.from_pretrained(BLIP_NAME).to(DEVICE)

    vqa_model.eval()
    vision_model.eval()

    verifier = None
    vocab = None
    config = None

    ckpt_path = ROOT / "outputs" / "checkpoints" / "verifier.pt"
    vocab_path = ROOT / "outputs" / "artifacts" / "vocab.json"
    config_path = ROOT / "outputs" / "artifacts" / "verifier_config.json"

    if ckpt_path.exists() and vocab_path.exists() and config_path.exists():
        vocab = load_json(str(vocab_path))
        config = load_json(str(config_path))

        verifier = MultimodalVerifier(
            vocab_size=len(vocab),
            image_dim=vision_model.config.vision_config.hidden_size,
            proj_dim=256,
            text_emb_dim=128,
            text_hidden_dim=128,
            dropout=0.3,
        ).to(DEVICE)

        state = torch.load(str(ckpt_path), map_location=DEVICE)
        verifier.load_state_dict(state)
        verifier.eval()

    return processor, vqa_model, vision_model, verifier, vocab, config


@torch.no_grad()
def generate_answer(processor, vqa_model, image, question):
    inputs = processor(images=image, text=question, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = vqa_model.generate(**inputs, max_new_tokens=10)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer.strip()


@torch.no_grad()
def extract_image_feature(processor, vision_model, image):
    proc = processor(images=image, return_tensors="pt")
    pixel_values = proc["pixel_values"].to(DEVICE)

    vision_out = vision_model.vision_model(pixel_values=pixel_values, return_dict=True)
    if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
        return vision_out.pooler_output
    return vision_out.last_hidden_state.mean(dim=1)


@torch.no_grad()
def verify_answer(processor, vision_model, verifier, vocab, config, image, question, answer):
    text = f"{question} [SEP] {answer}"
    ids = encode_text(text, vocab, config["max_len"])
    if len(ids) == 0:
        ids = [vocab["<unk>"]]

    input_ids = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    lengths = torch.tensor([len(ids)], dtype=torch.long).to(DEVICE)
    image_feat = extract_image_feature(processor, vision_model, image)

    logits = verifier(image_feat, input_ids, lengths).squeeze(-1)
    prob = torch.sigmoid(logits).item()
    label = "Likely Correct" if prob >= 0.5 else "Likely Incorrect"
    return label, prob


processor, vqa_model, vision_model, verifier, vocab, config = load_models()

st.title("Multimodal VQA with Answer Verification")
st.write("Upload an image, ask a question, generate an answer, and verify whether the answer is likely correct.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
question = st.text_input("Question", value="What is in the image?")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run"):
        answer = generate_answer(processor, vqa_model, image, question)
        st.subheader("Generated Answer")
        st.success(answer if answer else "[empty answer]")

        st.subheader("Verification")
        if verifier is None:
            st.warning("Verifier artifacts not found. Run training first.")
        else:
            label, prob = verify_answer(
                processor,
                vision_model,
                verifier,
                vocab,
                config,
                image,
                question,
                answer,
            )
            st.write(f"**Prediction:** {label}")
            st.write(f"**Confidence:** {prob:.4f}")
            st.progress(prob if prob >= 0.5 else 1.0 - prob)
else:
    st.info("Upload an image to begin.")