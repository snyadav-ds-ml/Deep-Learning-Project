import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BlipForQuestionAnswering, BlipProcessor

from utils import save_json, set_seed, token_f1, exact_match


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Salesforce/blip-vqa-base"


@torch.no_grad()
def generate_answer(processor, model, image, question: str, max_new_tokens: int = 10) -> str:
    inputs = processor(images=image, text=question, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer.strip()


def run_split(split: str, processor, model):
    df = pd.read_pickle(f"data/processed/{split}.pkl")

    predictions = []
    em_scores = []
    f1_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating {split} answers"):
        pred = generate_answer(processor, model, row["image"], row["question"])
        predictions.append(pred)
        em_scores.append(exact_match(row["answer"], pred))
        f1_scores.append(token_f1(row["answer"], pred))

    df["pred_answer"] = predictions
    df.to_pickle(f"outputs/predictions/{split}_with_pred.pkl")

    metrics = {
        "split": split,
        "num_examples": len(df),
        "exact_match": float(sum(em_scores) / len(em_scores)),
        "token_f1": float(sum(f1_scores) / len(f1_scores)),
    }
    save_json(metrics, f"outputs/metrics/generation_{split}.json")
    print(metrics)


def main():
    set_seed(42)
    os.makedirs("outputs/predictions", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    for split in ["train", "validation"]:
        run_split(split, processor, model)

    print("Answer generation complete.")


if __name__ == "__main__":
    main()