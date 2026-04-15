import pandas as pd
from utils import exact_match
from sklearn.metrics import accuracy_score

def main():
    df = pd.read_pickle("outputs/predictions/validation_with_pred.pkl")

    y_true = [1] * len(df)
    y_pred = [exact_match(g, p) for g, p in zip(df["answer"], df["pred_answer"])]

    acc = accuracy_score(y_true, y_pred)
    print("Generation Accuracy:", acc)

if __name__ == "__main__":
    main()