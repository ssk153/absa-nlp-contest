import pandas as pd
import numpy as np
import torch
import os
import argparse
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

LABEL_MAP = {
    0: 'none',
    1: 'positive',
    2: 'negative',
    3: 'neutral',
    4: 'conflict'
}
ASPECTS = ['food', 'anecdotes/miscellaneous', 'service', 'ambience', 'price']

class ABSATestDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def main():
    parser = argparse.ArgumentParser(description="Predict ABSA Model")
    parser.add_argument('--model_dir', type=str, default='./trained_absa_model_deberta-v3-base', help='Directory of the trained model')
    parser.add_argument('--test_file', type=str, default='data/contest2_test.csv' if os.path.exists('data/contest2_test.csv') else 'contest2_test.csv', help='Path to test data CSV')
    parser.add_argument('--output_file', type=str, default='outputs/predictions.csv', help='Output csv filename')
    args = parser.parse_args()

    test_file = args.test_file
    if not os.path.exists(test_file) and os.path.exists(os.path.join('data', test_file)):
        test_file = os.path.join('data', test_file)

    if not os.path.exists(test_file):
        print(f"File {test_file} not found.")
        return

    print("Loading test data...")
    df_test = pd.read_csv(test_file)
    
    # We will format inputs by repeating each test row for all 5 aspects
    texts = []
    aspects_list = []
    ids = []
    
    for _, row in df_test.iterrows():
        for aspect in ASPECTS:
            texts.append(row['text'])
            aspects_list.append(aspect)
            ids.append(row['id'])
            
    model_path = args.model_dir
    if not os.path.exists(model_path):
        print(f"Model path {model_path} not found. Please train the model first.")
        return
        
    print(f"Loading tokenizer and model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5)
    
    print("Tokenizing test data...")
    test_encodings = tokenizer(texts, aspects_list, truncation=True, padding=True, max_length=256)
    test_dataset = ABSATestDataset(test_encodings)
    
    # We can use Trainer for easy inference
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=32,
        do_predict=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
    )
    
    print("Generating predictions...")
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    predicted_classes = np.argmax(logits, axis=-1)
    
    print("Formatting predictions...")
    out_data = []
    
    # Process predictions per ID (every 5 items correspond to 1 ID)
    for i in range(0, len(predicted_classes), len(ASPECTS)):
        id_val = ids[i]
        text_val = texts[i]
        
        has_aspect = False
        best_aspect_idx = -1
        best_polarity = -1
        max_logit = -float('inf')
        
        for j in range(len(ASPECTS)):
            idx = i + j
            pred_cls = predicted_classes[idx]
            polarity = LABEL_MAP[pred_cls]
            
            # Find the best non-none logit in case we predict all none
            # Logits for classes 1, 2, 3, 4
            non_none_logits = logits[idx][1:]
            max_non_none_val = np.max(non_none_logits)
            max_non_none_cls = np.argmax(non_none_logits) + 1
            
            if max_non_none_val > max_logit:
                max_logit = max_non_none_val
                best_aspect_idx = j
                best_polarity = max_non_none_cls
                
            if polarity != 'none':
                has_aspect = True
                out_data.append({
                    'id': id_val,
                    'text': text_val,
                    'aspectCategory': aspects_list[idx],
                    'polarity': polarity
                })
                
        # If all aspects predicted none, force the most confident non-none aspect
        if not has_aspect:
            out_data.append({
                'id': id_val,
                'text': text_val,
                'aspectCategory': aspects_list[i + best_aspect_idx],
                'polarity': LABEL_MAP[best_polarity]
            })
            
    df_out = pd.DataFrame(out_data)
    out_file = args.output_file
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(out_file, index=False)
    print(f"Predictions saved to {out_file} (Total rows: {len(df_out)})")

if __name__ == "__main__":
    main()

