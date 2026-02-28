import ft_config
from ft_train import train_fasttext_model

def main() -> None:
    """
    given nothing
    return nothing
    executes fasttext training pipeline
    """
    train_fasttext_model(ft_config.TRAIN_FILE, ft_config.MODEL_FILE)

if __name__ == "__main__":
    main()
