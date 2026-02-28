import lgb_config
from lgb_train import train_lgbm_model

def main() -> None:
    """
    given nothing
    return nothing
    executes lightgbm training pipeline
    """
    train_lgbm_model(lgb_config.TRAIN_FILE, lgb_config.MODEL_FILE)

if __name__ == "__main__":
    main()
