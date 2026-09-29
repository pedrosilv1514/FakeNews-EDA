"""Interface de inferencia reutilizavel por CLI ou futuro backend FastAPI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.preprocessing.modeling import compose_news_text


class NewsClassifier:
    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path)
        artifact = joblib.load(self.model_path)
        if not isinstance(artifact, dict) or not {"estimator", "metadata"} <= set(artifact):
            raise ValueError("Artefato invalido: estimator e metadata sao obrigatorios.")
        self.estimator = artifact["estimator"]
        self.metadata = artifact["metadata"]

    def predict(self, *, title: str = "", content: str = "", subtitle: str = "") -> dict[str, object]:
        text = compose_news_text(title, content, subtitle)
        if len(text) < 20:
            raise ValueError("Forneca titulo ou conteudo textual suficiente para classificacao.")
        probability_fake = float(self.estimator.predict_proba([text])[0, 1])
        threshold = float(self.metadata.get("threshold", 0.5))
        predicted_class = "fake" if probability_fake >= threshold else "real"
        predicted_probability = probability_fake if predicted_class == "fake" else 1 - probability_fake
        return {
            "predicted_class": predicted_class,
            "predicted_class_probability": predicted_probability,
            "probability_fake": probability_fake,
            "probability_real": 1 - probability_fake,
            "threshold": threshold,
            "model_id": self.metadata.get("experiment", self.model_path.stem),
            "model_version": self.metadata.get("model_version", "unknown"),
            "disclaimer": self.metadata.get(
                "disclaimer",
                "Probabilidade de pertencimento a classe; nao comprova a veracidade factual.",
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/E02_linear_svm.joblib"))
    parser.add_argument("--title", default="")
    parser.add_argument("--content", default="")
    parser.add_argument("--subtitle", default="")
    args = parser.parse_args()
    predictor = NewsClassifier(args.model)
    print(json.dumps(predictor.predict(title=args.title, content=args.content, subtitle=args.subtitle), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
