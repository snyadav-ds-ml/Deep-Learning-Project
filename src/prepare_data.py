from datasets import load_dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def main():
    dataset = load_dataset("merve/vqav2-small")

    print(dataset)

    os.makedirs("data/processed", exist_ok=True)

    # only available split
    df = pd.DataFrame(dataset["validation"])

    # rename columns to keep code consistent
    df = df.rename(columns={"multiple_choice_answer": "answer"})

    df = df.sample(n=2000, random_state=42).reset_index(drop=True)

    # split into train/validation
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_pickle("data/processed/train.pkl")
    val_df.to_pickle("data/processed/validation.pkl")

    print(f"✅ Train size: {len(train_df)}")
    print(f"✅ Validation size: {len(val_df)}")
    print("✅ Data prepared successfully")

if __name__ == "__main__":
    main()