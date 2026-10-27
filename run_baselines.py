import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
import json
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
train_df = df_processed.iloc[train_idx].copy()
val_df = df_processed.iloc[val_idx].copy()

# Logistic Regression baseline
# Input text is just text + aspect
train_df['text_aspect'] = train_df['text'] + " " + train_df['aspect']
val_df['text_aspect'] = val_df['text'] + " " + val_df['aspect']

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_df['text_aspect'])
y_train = train_df['label']

X_val = vectorizer.transform(val_df['text_aspect'])
y_val = val_df['label']

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

# Compute same metric as their evaluate.py
# F1 where we consider correct if BOTH aspect and sentiment are correct.
# A true positive is a non-none prediction that matches a non-none ground truth for the same ID/aspect
val_df['pred_label'] = y_pred

# Save to temp csv and evaluate
out_data = []
for _, row in val_df.iterrows():
    if row['pred_label'] != 0:
        out_data.append({
            'id': row['id'],
            'text': row['text'],
            'aspectCategory': row['aspect'],
            'polarity': inv_map[row['pred_label']]
        })
os.makedirs('outputs', exist_ok=True)
pd.DataFrame(out_data).to_csv('outputs/val_set_predictions_lr.csv', index=False)

print("Logistic Regression run complete.")
