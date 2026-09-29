from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.predict import NewsClassifier
from src.preprocessing.modeling import compose_news_text


def test_model_can_be_loaded_and_used_for_inference(tmp_path: Path) -> None:
    texts = [
        compose_news_text("Notícia verificada", "Informação confirmada por documentos"),
        compose_news_text("Boato viral", "Mensagem falsa circula nas redes"),
        compose_news_text("Fato confirmado", "Dados oficiais foram publicados"),
        compose_news_text("É falso", "A alegação foi desmentida"),
    ]
    estimator = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(texts, [0, 1, 0, 1])
    path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "estimator": estimator,
            "metadata": {"experiment": "TEST", "model_version": "0", "threshold": 0.5},
        },
        path,
    )
    result = NewsClassifier(path).predict(title="Alegação nova", content="Texto suficiente para avaliar")
    assert result["predicted_class"] in {"fake", "real"}
    assert 0 <= result["probability_fake"] <= 1
    assert result["model_id"] == "TEST"
