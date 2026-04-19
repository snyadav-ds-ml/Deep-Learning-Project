import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return TOKEN_RE.findall(text)


def exact_match(a: str, b: str) -> int:
    return int(normalize_text(a) == normalize_text(b))


def token_f1(a: str, b: str) -> float:
    gold = tokenize(a)
    pred = tokenize(b)

    if len(gold) == 0 and len(pred) == 0:
        return 1.0
    if len(gold) == 0 or len(pred) == 0:
        return 0.0

    gold_counter = Counter(gold)
    pred_counter = Counter(pred)
    common = gold_counter & pred_counter
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def build_vocab(texts: List[str], min_freq: int = 1, max_size: int = 20000) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        vocab[token] = len(vocab)
        if len(vocab) >= max_size:
            break
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 40) -> List[int]:
    tokens = tokenize(text)[:max_len]
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokens]


def build_verification_examples(df, seed: int = 42):
    """
    Builds positive and negative examples for the verifier.

    Positive:
      - (image, question, gold_answer) -> 1

    Negatives:
      - (image, question, pred_answer) -> 0 if pred != gold
      - (image, question, random_other_answer) -> 0
    """
    rng = random.Random(seed)
    all_answers = [str(x) for x in df["answer"].tolist()]
    examples = []

    for _, row in df.iterrows():
        image = row["image"]
        question = str(row["question"])
        gold = str(row["answer"])
        pred = str(row.get("pred_answer", ""))

        # positive
        examples.append(
            {
                "image": image,
                "question": question,
                "candidate": gold,
                "label": 1,
            }
        )

        # hard negative from generated answer
        if normalize_text(pred) and normalize_text(pred) != normalize_text(gold):
            examples.append(
                {
                    "image": image,
                    "question": question,
                    "candidate": pred,
                    "label": 0,
                }
            )

        # random negative
        wrong = rng.choice(all_answers)
        while normalize_text(wrong) == normalize_text(gold):
            wrong = rng.choice(all_answers)

        examples.append(
            {
                "image": image,
                "question": question,
                "candidate": wrong,
                "label": 0,
            }
        )

    return examples


def save_json(obj, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)