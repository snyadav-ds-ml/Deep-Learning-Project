import random

def normalize(text):
    return text.lower().strip()

def exact_match(a, b):
    return int(normalize(a) == normalize(b))

def build_verification_examples(df):
    examples = []
    answers = df["answer"].tolist()

    for _, row in df.iterrows():
        image = row["image"]
        question = row["question"]
        gold = row["answer"]
        pred = row["pred_answer"]

        # positive
        examples.append({
            "image": image,
            "question": question,
            "candidate": gold,
            "label": 1
        })

        # negative: wrong pred
        if normalize(pred) != normalize(gold):
            examples.append({
                "image": image,
                "question": question,
                "candidate": pred,
                "label": 0
            })

        # negative: random answer
        wrong = random.choice(answers)
        while normalize(wrong) == normalize(gold):
            wrong = random.choice(answers)

        examples.append({
            "image": image,
            "question": question,
            "candidate": wrong,
            "label": 0
        })

    return examples