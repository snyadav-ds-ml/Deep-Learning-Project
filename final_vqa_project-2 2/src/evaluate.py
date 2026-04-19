import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BlipModel, BlipProcessor

from models import MultimodalVerifier
from utils import (
    build_verification_examples,
    encode_text,
    exact_match,
    load_json,
    token_f1,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VerificationDataset(Dataset):
    def __init__(self, examples, vocab, max_len=40):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = f"{ex['question']} [SEP] {ex['candidate']}"
        ids = encode_text(text, self.vocab, self.max_len)
        if len(ids) == 0:
            ids = [self.vocab["<unk>"]]

        return {
            "image": ex["image"],
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": len(ids),
            "label": float(ex["label"]),
        }


def collate_fn(batch):
    images = [x["image"] for x in batch]
    input_ids = [x["input_ids"] for x in batch]
    lengths = torch.tensor([x["length"] for x in batch], dtype=torch.long)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)

    padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    return images, padded, lengths, labels


@torch.no_grad()
def extract_image_features(images, processor, vision_model):
    proc = processor(images=images, return_tensors="pt")
    pixel_values = proc["pixel_values"].to(DEVICE)

    vision_out = vision_model.vision_model(pixel_values=pixel_values, return_dict=True)
    if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
        feats = vision_out.pooler_output
    else:
        feats = vision_out.last_hidden_state.mean(dim=1)

    return feats


def evaluate_generation():
    df = pd.read_pickle("outputs/predictions/validation_with_pred.pkl")

    em = [exact_match(g, p) for g, p in zip(df["answer"], df["pred_answer"])]
    f1s = [token_f1(g, p) for g, p in zip(df["answer"], df["pred_answer"])]

    print("=== GENERATION RESULTS ===")
    print(f"Validation Exact Match : {sum(em) / len(em):.4f}")
    print(f"Validation Token F1    : {sum(f1s) / len(f1s):.4f}")


def evaluate_verifier():
    df = pd.read_pickle("outputs/predictions/validation_with_pred.pkl")
    vocab = load_json("outputs/artifacts/vocab.json")
    config = load_json("outputs/artifacts/verifier_config.json")

    processor = BlipProcessor.from_pretrained(config["blip_name"])
    vision_model = BlipModel.from_pretrained(config["blip_name"]).to(DEVICE)
    vision_model.eval()

    image_dim = vision_model.config.vision_config.hidden_size

    model = MultimodalVerifier(
        vocab_size=len(vocab),
        image_dim=image_dim,
        proj_dim=256,
        text_emb_dim=128,
        text_hidden_dim=128,
        dropout=0.3,
    ).to(DEVICE)

    state = torch.load("outputs/checkpoints/verifier.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    examples = build_verification_examples(df, seed=42)
    dataset = VerificationDataset(examples, vocab, max_len=config["max_len"])
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, input_ids, lengths, labels in loader:
            feats = extract_image_features(images, processor, vision_model)
            logits = model(feats, input_ids.to(DEVICE), lengths.to(DEVICE)).squeeze(-1)
            probs = torch.sigmoid(logits)

            y_true.extend(labels.long().tolist())
            y_pred.extend((probs >= 0.5).long().cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    print("\n=== VERIFIER RESULTS ===")
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1        : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    if len(set(y_true)) > 1:
        print(f"AUC       : {roc_auc_score(y_true, y_prob):.4f}")


def main():
    evaluate_generation()
    evaluate_verifier()


if __name__ == "__main__":
    main()