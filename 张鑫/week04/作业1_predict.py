"""
Evaluate and preview predictions for checkpoints saved under outputs_agnews.

Usage:
    python 作业1_predict.py [--limit N] [--ckpt checkpoint-XX]

Assumptions:
    - Train/Test CSVs: ag_news_train.csv, ag_news_test.csv (columns: text,label)
    - Checkpoints live in: outputs_agnews/checkpoint-*/ (tokenizer + model)
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer


BASE = Path(__file__).resolve().parent
TRAIN_CSV = BASE / "ag_news_train.csv"
TEST_CSV = BASE / "ag_news_test.csv"
CKPT_ROOT = BASE / "outputs_agnews"
MAX_LEN = 64
BATCH = 128


def load_label_encoder() -> LabelEncoder:
    df = pd.read_csv(TRAIN_CSV)
    lbl = LabelEncoder().fit(df["label"].values)
    return lbl


def make_loader(tokenizer: BertTokenizer, texts: List[str], labels: List[int]) -> DataLoader:
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(labels))
    return DataLoader(ds, batch_size=BATCH)


@torch.no_grad()
def eval_ckpt(ckpt: Path, lbl: LabelEncoder, test_texts: List[str], test_labels: List[int]):
    if not ckpt.exists():
        print(f"skip {ckpt.name}: not found")
        return

    tokenizer = BertTokenizer.from_pretrained(str(ckpt))
    model = BertForSequenceClassification.from_pretrained(ckpt).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_loader = make_loader(tokenizer, test_texts, test_labels)

    correct, total = 0, 0
    for input_ids, attention_mask, labels in test_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / total if total else 0.0
    print(f"{ckpt.name}: accuracy={acc:.4f} ({correct}/{total})")

    # preview first 5 predictions
    sample_texts = pd.read_csv(TEST_CSV).head(5)
    enc = tokenizer(
        sample_texts["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)
    sample_logits = model(**enc).logits
    sample_preds = sample_logits.argmax(dim=-1).cpu().numpy()
    classes = lbl.classes_
    print("  sample predictions:")
    for t, p in zip(sample_texts["text"].tolist(), sample_preds):
        print(f"    {classes[p]} | {t[:60]}...")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only evaluate first N test samples")
    parser.add_argument("--ckpt", type=str, default=None, help="evaluate a single checkpoint folder name, e.g., checkpoint-20")
    args = parser.parse_args()

    lbl = load_label_encoder()
    test_df = pd.read_csv(TEST_CSV)
    if args.limit:
        test_df = test_df.head(args.limit)
    test_texts = test_df["text"].tolist()
    test_labels = lbl.transform(test_df["label"].values)

    ckpts = sorted([p for p in CKPT_ROOT.glob("checkpoint-*") if p.is_dir()],
                   key=lambda p: int(p.name.split("-")[-1]))
    if args.ckpt:
        target = CKPT_ROOT / args.ckpt
        ckpts = [target] if target.is_dir() else []
    if not ckpts:
        print(f"no checkpoint found under: {CKPT_ROOT}")
        return
    for ckpt in ckpts:
        eval_ckpt(ckpt, lbl, test_texts, test_labels)


if __name__ == "__main__":
    main()
