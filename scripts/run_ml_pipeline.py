"""Executa preparacao, baselines, stacking e avaliacao temporal."""

from src.evaluate_temporal import evaluate_temporal
from src.preprocessing.modeling import prepare_modeling_dataset
from src.train_baselines import train_baselines
from src.train_stacking import train_stacking


if __name__ == "__main__":
    prepare_modeling_dataset()
    train_baselines()
    train_stacking()
    evaluate_temporal()
