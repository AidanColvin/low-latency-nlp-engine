"""
holds the full cross-validation, training, and prediction pipeline
for comparing multiple deep learning models
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

# --- Configuration ---
TRAIN_PATH = "data/processed/train.tsv"
TEST_PATH = "data/raw/test.tsv"
FOLDS = 3
EPOCHS = 1
BATCH_SIZE = 16

MODELS = {
    "TinyBERT": "huawei-noah/TinyBERT_General_4L_312D",
    "ALBERT": "albert-base-v2",
    "RoBERTa": "roberta-base"
}

class TextDataset(torch.utils.data.Dataset):
    """
    given dictionary of encodings and optional list of labels
    return pytorch dataset object
    """
    def __init__(self, encodings: dict, labels: list[int] = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.encodings['input_ids'])

def load_data(train_file: str, test_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    given file paths for train and test sets
    return tuple of pandas dataframes
    drops missing values from training set
    """
    train_df = pd.read_csv(train_file, sep='\t').dropna(subset=['review', 'label'])
    test_df = pd.read_csv(test_file, sep='\t')
    return train_df, test_df

def compute_metrics(pred) -> dict:
    """
    given transformer predictions
    return dictionary with accuracy score
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def run_model_pipeline(model_name: str, hf_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame, n_splits: int) -> float:
    """
    given model configurations and dataframes
    return mean accuracy float across folds
    trains model, evaluates, and generates formatted submission file
    """
    print(f"\n========== Starting {model_name} ==========")
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    
    # Encode test data once
    test_encodings = tokenizer(test_df['review'].tolist(), truncation=True, padding=True, max_length=128)
    test_dataset = TextDataset(test_encodings)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    test_predictions = []
    
    texts = np.array(train_df['review'].tolist())
    labels = np.array(train_df['label'].tolist())
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
        print(f"\n--- {model_name} : Fold {fold + 1}/{n_splits} ---")
        
        train_texts, val_texts = texts[train_idx].tolist(), texts[val_idx].tolist()
        train_labels, val_labels = labels[train_idx].tolist(), labels[val_idx].tolist()
        
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
        
        train_dataset = TextDataset(train_encodings, train_labels)
        val_dataset = TextDataset(val_encodings, val_labels)
        
        model = AutoModelForSequenceClassification.from_pretrained(hf_path, num_labels=2)
        
        training_args = TrainingArguments(
            output_dir=f'./results_{model_name}_f{fold}',
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            evaluation_strategy="epoch",
            logging_steps=50,
            save_strategy="none"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train and evaluate
        trainer.train()
        eval_res = trainer.evaluate()
        fold_accuracies.append(eval_res['eval_accuracy'])
        print(f"Fold {fold + 1} Accuracy: {eval_res['eval_accuracy']:.4f}")
        
        # Predict on Test set
        preds = trainer.predict(test_dataset)
        test_predictions.append(preds.predictions)
        
    # Average predictions across all folds
    avg_preds = np.mean(test_predictions, axis=0)
    final_labels = np.argmax(avg_preds, axis=-1)
    
    # Generate requested submission file
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'label': final_labels
    })
    output_filename = f"{model_name}_submision_{n_splits}_version_1.csv"
    submission_df.to_csv(output_filename, index=False)
    print(f"Saved predictions to {output_filename}")
    
    mean_acc = np.mean(fold_accuracies)
    return mean_acc

def main() -> None:
    """
    given nothing
    return nothing
    orchestrates data loading, model comparisons, and outputs final accuracy table
    """
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    
    results = {}
    for name, path in MODELS.items():
        mean_acc = run_model_pipeline(name, path, train_df, test_df, FOLDS)
        results[name] = mean_acc
        
    print("\n" + "="*40)
    print("FINAL ACCURACY COMPARISON (Cross-Validation Mean)")
    print("="*40)
    for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name}: {acc:.4f}")

if __name__ == "__main__":
    main()
