import pandas as pd
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from tqdm import tqdm
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(DEVICE)
model.eval()

def generate_answer(image, question):
    inputs = processor(images=image, text=question, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

def main():
    os.makedirs("outputs/predictions", exist_ok=True)

    for split in ["train", "validation"]:
        df = pd.read_pickle(f"data/processed/{split}.pkl")

        preds = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            ans = generate_answer(row["image"], row["question"])
            preds.append(ans)

        df["pred_answer"] = preds
        df.to_pickle(f"outputs/predictions/{split}_with_pred.pkl")

    print("✅ Answers generated")

if __name__ == "__main__":
    main()