import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BlipProcessor, BlipModel
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import build_verification_examples
from models import Verifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip = BlipModel.from_pretrained("Salesforce/blip-vqa-base").to(DEVICE)
blip.eval()

class VDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate(batch):
    images = [x["image"] for x in batch]
    texts = [x["question"] + " " + x["candidate"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)

    proc = processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    return proc, labels

def get_emb(proc):
    proc = {k: v.to(DEVICE) for k, v in proc.items()}

    with torch.no_grad():
        out = blip(**proc, return_dict=True)

        img_emb = out.image_embeds
        txt_emb = out.text_embeds

        # Safety for version differences
        if hasattr(img_emb, "last_hidden_state"):
            img_emb = img_emb.last_hidden_state
        if hasattr(txt_emb, "last_hidden_state"):
            txt_emb = txt_emb.last_hidden_state

        if img_emb.dim() == 3:
            img_emb = img_emb.mean(dim=1)
        if txt_emb.dim() == 3:
            txt_emb = txt_emb.mean(dim=1)

    return img_emb, txt_emb

def evaluate_model(model, val_loader):
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for proc, labels in val_loader:
            img, txt = get_emb(proc)
            logits = model(img, txt).squeeze(-1)
            pred = (torch.sigmoid(logits) > 0.5).int().cpu().tolist()

            preds.extend(pred)
            true.extend(labels.int().tolist())

    acc = accuracy_score(true, preds)
    prec = precision_score(true, preds, zero_division=0)
    rec = recall_score(true, preds, zero_division=0)
    f1 = f1_score(true, preds, zero_division=0)

    return acc, prec, rec, f1

def main():
    os.makedirs("outputs/checkpoints", exist_ok=True)

    train_df = pd.read_pickle("outputs/predictions/train_with_pred.pkl")
    val_df = pd.read_pickle("outputs/predictions/validation_with_pred.pkl")

    train_data = build_verification_examples(train_df)
    val_data = build_verification_examples(val_df)

    train_loader = DataLoader(
        VDataset(train_data),
        batch_size=8,
        shuffle=True,
        collate_fn=collate
    )
    val_loader = DataLoader(
        VDataset(val_data),
        batch_size=8,
        shuffle=False,
        collate_fn=collate
    )

    model = Verifier(emb_dim=512).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_f1 = -1

    for epoch in range(3):
        model.train()
        total_loss = 0.0

        for proc, labels in tqdm(train_loader):
            labels = labels.to(DEVICE)

            img, txt = get_emb(proc)
            logits = model(img, txt).squeeze(-1)
            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        acc, prec, rec, f1 = evaluate_model(model, val_loader)
        print(
            f"Epoch {epoch+1} | "
            f"loss={total_loss/len(train_loader):.4f} | "
            f"acc={acc:.4f} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "outputs/checkpoints/verifier.pt")

    print("✅ Verifier trained and saved")

if __name__ == "__main__":
    main()