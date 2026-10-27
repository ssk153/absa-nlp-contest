import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

LABEL_MAP = {
    0: 'none',
    1: 'positive',
    2: 'negative',
    3: 'neutral',
    4: 'conflict'
}
ASPECTS = ['food', 'anecdotes/miscellaneous', 'service', 'ambience', 'price']

class ABSADataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

def process_and_save_predictions(trainer, dataset, df_split, out_filename):
    print(f"Generating predictions for {out_filename}...")
    preds = trainer.predict(dataset)
    predicted_classes = np.argmax(preds.predictions, axis=-1)
    
    out_data = []
    for i, pred_cls in enumerate(predicted_classes):
        polarity = LABEL_MAP[pred_cls]
        if polarity != 'none':
            out_data.append({
                'id': df_split.iloc[i]['id'],
                'text': df_split.iloc[i]['text'],
                'aspectCategory': df_split.iloc[i]['aspect'],
                'polarity': polarity
            })
            
    df_out = pd.DataFrame(out_data)
    df_out.to_csv(out_filename, index=False)
    print(f"Saved {len(df_out)} predictions to {out_filename}")
    
def save_ground_truth(df_split, out_filename):
    out_data = []
    for _, row in df_split.iterrows():
        polarity_val = row['label']
        if polarity_val != 0: # 0 is 'none'
            out_data.append({
                'id': row['id'],
                'text': row['text'],
                'aspectCategory': row['aspect'],
                'polarity': LABEL_MAP[polarity_val]
            })
    df_out = pd.DataFrame(out_data)
    df_out.to_csv(out_filename, index=False)
    print(f"Saved {len(df_out)} ground truth rows to {out_filename}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict Train/Val Model")
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-large', choices=['deberta-v3-base', 'microsoft/deberta-v3-large', 'roberta-large'])
    args = parser.parse_args()

    train_file = 'data/contest2_train.csv' if os.path.exists('data/contest2_train.csv') else 'contest2_train.csv'
    print(f"Loading data from {train_file}...")
    df = pd.read_csv(train_file)
    
    # Process data exactly as in train.py but include 'id'
    data = []
    # Note: we need to recreate the EXACT data list to ensure random split is identical
    for id_val, group in df.groupby('id'):
        text = group['text'].iloc[0]
        aspect_dict = dict(zip(group['aspectCategory'], group['polarity']))
        
        for aspect in ASPECTS:
            polarity = aspect_dict.get(aspect, 'none')
            # 0 to 4 mapping
            inv_map = {v: k for k, v in LABEL_MAP.items()}
            data.append({
                'id': id_val,
                'text': text,
                'aspect': aspect,
                'label': inv_map[polarity]
            })

    df_processed = pd.DataFrame(data)
    
    # Train/validation split with exact same seed as train.py
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(df_processed, groups=df_processed['id']))
    
    train_df = df_processed.iloc[train_idx]
    val_df = df_processed.iloc[val_idx]
    
    model_dir_name = args.model_name.replace('/', '_')
    model_path = f"./trained_absa_model_{model_dir_name}"
    print(f"Loading tokenizer and model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5)
    
    print("Tokenizing data...")
    train_encodings = tokenizer(train_df['text'].tolist(), train_df['aspect'].tolist(), truncation=True, padding=True, max_length=256)
    val_encodings = tokenizer(val_df['text'].tolist(), val_df['aspect'].tolist(), truncation=True, padding=True, max_length=256)
    
    train_dataset = ABSADataset(train_encodings)
    val_dataset = ABSADataset(val_encodings)
    
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=32,
        do_predict=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
    )
    
    # Process and save predictions
    process_and_save_predictions(trainer, train_dataset, train_df, f'train_set_predictions_{model_dir_name}.csv')
    process_and_save_predictions(trainer, val_dataset, val_df, f'val_set_predictions_{model_dir_name}.csv')
    
    # Save corresponding ground truth (since the original evaluate.py needs a ground truth file matching the IDs)
    # The user can run: python evaluate.py train_ground_truth.csv train_set_predictions.csv
    save_ground_truth(train_df, f'train_ground_truth_{model_dir_name}.csv')
    save_ground_truth(val_df, f'val_ground_truth_{model_dir_name}.csv')

if __name__ == "__main__":
    main()
