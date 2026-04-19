import os

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from utils import save_json, set_seed


def main():
    set_seed(42)
    os.makedirs("data/processed", exist_ok=True)

    dataset = load_dataset("merve/vqav2-small")
    print(dataset)

    # This small dataset mainly exposes validation, so we split it ourselves
    df = pd.DataFrame(dataset["validation"])

    # Keep naming consistent
    if "multiple_choice_answer" in df.columns:
        df = df.rename(columns={"multiple_choice_answer": "answer"})

    # Keep only columns we need
    keep_cols = [c for c in ["image", "question", "answer", "question_type", "answer_type"] if c in df.columns]
    df = df[keep_cols].copy()

    # Remove missing values
    df = df.dropna(subset=["image", "question", "answer"]).reset_index(drop=True)

    # Sample manageable subset
    n_samples = min(2000, len(df))
    df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_pickle("data/processed/train.pkl")
    val_df.to_pickle("data/processed/validation.pkl")

    metadata = {
        "dataset_name": "merve/vqav2-small",
        "total_examples": len(df),
        "train_size": len(train_df),
        "validation_size": len(val_df),
        "task": "Visual Question Answering with answer verification",
    }
    save_json(metadata, "outputs/metadata/data_summary.json")

    print(f"Saved train.pkl with {len(train_df)} rows")
    print(f"Saved validation.pkl with {len(val_df)} rows")
    print("Data preparation complete.")


if __name__ == "__main__":
    main()