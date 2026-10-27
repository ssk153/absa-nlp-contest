import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Labels
LABEL_MAP = {
    'none': 0,
    'positive': 1,
    'negative': 2,
    'neutral': 3,
    'conflict': 4
}
ASPECTS = ['food', 'anecdotes/miscellaneous', 'service', 'ambience', 'price']

class ABSADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1_macro": f1}

class CustomTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Move class_weights to the same device as logits
        self.class_weights = self.class_weights.to(logits.device)
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def main():
    parser = argparse.ArgumentParser(description="Train ABSA Model")
    parser.add_argument('--model_name', type=str, default='microsoft/deberta-v3-large', help='Model to use')
    parser.add_argument('--train_file', type=str, default='data/contest2_train.csv' if os.path.exists('data/contest2_train.csv') else 'contest2_train.csv', help='Path to training data CSV')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    args = parser.parse_args()

    train_file = args.train_file
    if not os.path.exists(train_file) and os.path.exists(os.path.join('data', train_file)):
        train_file = os.path.join('data', train_file)

    print(f"Loading data from {train_file}...")
    df = pd.read_csv(train_file)
    
    # Process data to create a sequence classification dataset
    data = []
    for id_val, group in df.groupby('id'):
        text = group['text'].iloc[0]
        aspect_dict = dict(zip(group['aspectCategory'], group['polarity']))
        
        for aspect in ASPECTS:
            polarity = aspect_dict.get(aspect, 'none')
            data.append({
                'id': id_val,
                'text': text,
                'aspect': aspect,
                'label': LABEL_MAP[polarity]
            })

    df_processed = pd.DataFrame(data)
    
    # GroupShuffleSplit ensures that all rows with the same 'id' end up in the SAME split
    print("Splitting data based on ID to prevent data leakage...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, val_idx = next(gss.split(df_processed, groups=df_processed['id']))
    
    train_df = df_processed.iloc[train_idx]
    val_df = df_processed.iloc[val_idx]
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")

    # Compute class weights
    labels_train = train_df['label'].tolist()
    unique_classes = np.unique(labels_train)
    class_weights_array = compute_class_weight(class_weight='balanced', classes=unique_classes, y=labels_train)
    print(f"Computed class weights: {class_weights_array}")
    
    # Convert local path or use HF hub
    model_dir_name = args.model_name.replace('/', '_')
    model_path = f"./models/{model_dir_name}"
    
    if not os.path.exists(model_path):
        model_path = args.model_name 

    print(f"Loading tokenizer and model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Tokenization
    train_encodings = tokenizer(train_df['text'].tolist(), train_df['aspect'].tolist(), truncation=True, padding=True, max_length=args.max_length)
    val_encodings = tokenizer(val_df['text'].tolist(), val_df['aspect'].tolist(), truncation=True, padding=True, max_length=args.max_length)
    
    train_labels = train_df['label'].tolist()
    val_labels = val_df['label'].tolist()
    
    train_dataset = ABSADataset(train_encodings, train_labels)
    val_dataset = ABSADataset(val_encodings, val_labels)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5)
    
    training_args = TrainingArguments(
        output_dir=f"./results_{model_dir_name}",
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1
    )
    
    trainer = CustomTrainer(
        class_weights=class_weights_array,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    print(f"Starting training for {args.model_name}...")
    trainer.train()
    
    output_dir = f"./trained_absa_model_{model_dir_name}"
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete.")

if __name__ == "__main__":
    main()
