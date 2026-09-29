"""Experimento opcional E05 com BERTimbau Base.

Requer a instalacao do extra ``bert`` e, na pratica, GPU/MPS com memoria
suficiente. O script nunca e chamado pelo pipeline tradicional.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.preprocessing.modeling import prepare_modeling_dataset


MODEL_NAME = "neuralmind/bert-base-portuguese-cased"


def train_bert(output_dir: Path, epochs: int = 3, max_length: int = 512) -> None:
    try:
        import torch
        from datasets import Dataset
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            'Dependencias opcionais ausentes. Instale com: pip install -e ".[bert]"'
        ) from exc

    cfg = load_config()
    frame, _ = prepare_modeling_dataset()
    train = frame[frame["split"] == "train"][["text", "label"]]
    validation = frame[frame["split"] == "validation"][["text", "label"]]
    test = frame[frame["split"] == "test"][["record_id", "text", "label"]]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    datasets = {
        "train": Dataset.from_pandas(train, preserve_index=False).map(tokenize, batched=True),
        "validation": Dataset.from_pandas(validation, preserve_index=False).map(tokenize, batched=True),
        "test": Dataset.from_pandas(test[["text", "label"]], preserve_index=False).map(tokenize, batched=True),
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "real", 1: "fake"}, label2id={"real": 0, "fake": 1}
    )

    def compute_metrics(prediction):
        labels = prediction.label_ids
        predicted = np.argmax(prediction.predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predicted),
            "precision": precision_score(labels, predicted, zero_division=0),
            "recall": recall_score(labels, predicted, zero_division=0),
            "f1": f1_score(labels, predicted, zero_division=0),
            "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=int(cfg["seed"]),
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_output = trainer.predict(datasets["test"])
    logits = test_output.predictions
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    predicted = (probabilities >= 0.5).astype(int)
    predictions = pd.DataFrame(
        {
            "record_id": test["record_id"].to_numpy(),
            "y_true": test["label"].to_numpy(),
            "y_pred": predicted,
            "probability_fake": probabilities,
        }
    )
    predictions.to_csv(cfg["project_root"] / "reports/metrics/E05_test_predictions.csv", index=False)
    trainer.save_model(output_dir / "best_model")
    tokenizer.save_pretrained(output_dir / "best_model")
    (output_dir / "run_metadata.json").write_text(
        json.dumps({"base_model": MODEL_NAME, "epochs": epochs, "max_length": max_length}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models/E05_bertimbau"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    train_bert(args.output_dir, args.epochs, args.max_length)


if __name__ == "__main__":
    main()
