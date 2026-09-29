from src.data.generate_dataset import main as generate_dataset
from src.models.train import main as train_model


if __name__ == "__main__":
    generate_dataset()
    train_model()
