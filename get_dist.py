import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os

train_file = 'data/contest2_train.csv' if os.path.exists('data/contest2_train.csv') else 'contest2_train.csv'
df = pd.read_csv(train_file)
ASPECTS = ['food', 'anecdotes/miscellaneous', 'service', 'ambience', 'price']
LABEL_MAP = {
    'none': 0,
    'positive': 1,
    'negative': 2,
    'neutral': 3,
    'conflict': 4
}
inv_map = {v: k for k, v in LABEL_MAP.items()}

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
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(gss.split(df_processed, groups=df_processed['id']))
train_df = df_processed.iloc[train_idx]
val_df = df_processed.iloc[val_idx]

# Map back to string labels for counting (ignoring 'none' if we want just the active labels)
print("--- TRAIN SET ---")
print(train_df[train_df['label'] != 0]['label'].map(inv_map).value_counts())
print("\n--- VAL SET ---")
print(val_df[val_df['label'] != 0]['label'].map(inv_map).value_counts())
