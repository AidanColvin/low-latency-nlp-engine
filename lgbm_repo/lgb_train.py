import lightgbm as lgb
import pandas as pd

def train_lgbm_model(data_path: str, model_path: str) -> None:
    """
    given a training data path and output model path
    return nothing
    trains gradient boosting tree model
    saves text-based model to disk
    """
    df = pd.read_csv(data_path, sep='\t').dropna()
    train_data = lgb.Dataset(df.drop('label', axis=1), label=df['label'])
    
    params = {'objective': 'binary', 'metric': 'binary_logloss'}
    bst = lgb.train(params, train_data, num_boost_round=100)
    bst.save_model(model_path)
