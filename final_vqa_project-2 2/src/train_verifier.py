import os
from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BlipModel, BlipProcessor

from models import MultimodalVerifier
from utils import (
    build_verification_examples,
    build_vocab,
    encode_text,
    load_json,
    save_json,
    set_seed,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLIP_NAME = "Salesforce/blip-vqa-base"
MAX_LEN = 40


@dataclass
class Batch:
    images: list
    input_ids: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor


class VerificationDataset(Dataset):
    def __init__(self, examples: List[dict], vocab: dict, max_len: int = 40):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = f"{ex['question']} [SEP] {ex['candidate']}"
        ids = encode_text(text, self.vocab, max_len=self.max_len)
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

    return Batch(
        images=images,
        input_ids=padded,
        lengths=lengths,
        labels=labels,
    )


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


@torch.no_grad()
def evaluate(model, loader, processor, vision_model):
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []

    for batch in loader:
        image_feats = extract_image_features(batch.images, processor, vision_model)
        input_ids = batch.input_ids.to(DEVICE)
        lengths = batch.lengths.to(DEVICE)

        logits = model(image_feats, input_ids, lengths).squeeze(-1)
        probs = torch.sigmoid(logits)

        preds = (probs >= 0.5).long().cpu().tolist()
        probs = probs.cpu().tolist()
        labels = batch.labels.long().cpu().tolist()

        y_true.extend(labels)
        y_pred.extend(preds)
        y_prob.extend(probs)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if len(set(y_true)) > 1:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["auc"] = None

    return metrics


def main():
    set_seed(42)

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/artifacts", exist_ok=True)

    train_df = pd.read_pickle("outputs/predictions/train_with_pred.pkl")
    val_df = pd.read_pickle("outputs/predictions/validation_with_pred.pkl")

    train_examples = build_verification_examples(train_df, seed=42)
    val_examples = build_verification_examples(val_df, seed=42)

    train_texts = [f"{x['question']} [SEP] {x['candidate']}" for x in train_examples]
    vocab = build_vocab(train_texts, min_freq=1, max_size=20000)
    save_json(vocab, "outputs/artifacts/vocab.json")

    train_dataset = VerificationDataset(train_examples, vocab, max_len=MAX_LEN)
    val_dataset = VerificationDataset(val_examples, vocab, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    processor = BlipProcessor.from_pretrained(BLIP_NAME)
    vision_model = BlipModel.from_pretrained(BLIP_NAME).to(DEVICE)
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

    pos_count = sum(ex["label"] for ex in train_examples)
    neg_count = len(train_examples) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_f1 = -1.0
    history = []

    for epoch in range(1, 6):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/5"):
            image_feats = extract_image_features(batch.images, processor, vision_model)
            input_ids = batch.input_ids.to(DEVICE)
            lengths = batch.lengths.to(DEVICE)
            labels = batch.labels.to(DEVICE)

            logits = model(image_feats, input_ids, lengths).squeeze(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        val_metrics = evaluate(model, val_loader, processor, vision_model)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = epoch_loss / max(len(train_loader), 1)
        history.append(val_metrics)

        print(val_metrics)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), "outputs/checkpoints/verifier.pt")

    save_json(history, "outputs/metrics/verifier_history.json")
    save_json(
        {
            "max_len": MAX_LEN,
            "blip_name": BLIP_NAME,
            "best_val_f1": best_f1,
        },
        "outputs/artifacts/verifier_config.json",
    )

    print(f"Best verifier F1: {best_f1:.4f}")
    print("Verifier training complete.")


if __name__ == "__main__":
    main()