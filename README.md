# Aspect-Based Sentiment Analysis

An end-to-end Natural Language Processing system for Aspect-Based Sentiment Analysis (ABSA) on restaurant reviews. The project formulates ABSA as a sequence-pair cross-encoder classification task and fine-tunes transformer models including RoBERTa-Large and DeBERTa-v3 to jointly detect aspect categories and classify their sentiment polarity.

## Results

The models were evaluated on a held-out validation split using the official competition metrics.

| Model | Overall Micro F1 | Aspect Macro F1 | Sentiment Macro F1 |
| :--- | :---: | :---: | :---: |
| **RoBERTa-large** | **0.751** | **0.935** | **0.582** |
| DeBERTa-v3-base | 0.735 | 0.927 | 0.544 |
| Zero-shot LLM Classifier | 0.610 | 0.725 | 0.630 |
| Logistic Regression (BoW) | 0.425 | 0.510 | 0.485 |

RoBERTa-large achieved the best overall Micro F1 score and was used for the final submission.

## Task

Given a customer review, the model identifies which predefined aspect categories are discussed and assigns a sentiment polarity to each identified aspect.

### Aspect categories

- `food`
- `service`
- `ambience`
- `price`
- `anecdotes/miscellaneous`

### Sentiment labels

- `positive`
- `negative`
- `neutral`
- `conflict`

## Methodology

### Sequence-Pair Cross-Encoder

Rather than using separate models for aspect extraction and sentiment classification, the project formulates the task as a sequence-pair classification problem similar to Natural Language Inference.

Each review is paired with each of the five target aspects:

```text
[CLS] <Review Text> [SEP] <Target Aspect> [SEP]
```

The model predicts one of five classes:

- `0`: none — the aspect is not mentioned
- `1`: positive
- `2`: negative
- `3`: neutral
- `4`: conflict

This formulation allows the transformer to use the relationship between the review and the queried aspect when determining sentiment.

### Grouped Validation Split

Reviews can contain multiple aspects, so splitting individual rows can cause the same review to appear in both training and validation sets. To prevent this leakage, `GroupShuffleSplit` is used with the review `id` as the grouping key and a 10% validation split. All aspect pairs belonging to the same review remain in the same split.

### Class-Weighted Loss

The sentiment classes are imbalanced:

- Positive: approximately 59.4%
- Negative: approximately 22.7%
- Neutral: approximately 12.6%
- Conflict: approximately 5.3%

Balanced class weights are applied to the cross-entropy loss through a custom trainer to reduce the effect of the class imbalance.

### Inference Fallback

During test inference, if all five aspect predictions are classified as `none`, the system selects the aspect-sentiment pair with the highest non-`none` logit confidence. This ensures that every test instance produces a valid prediction in the required submission format.

## Repository Structure

```text
.
├── data/
│   ├── contest2_train.csv                # Training data
│   └── contest2_test.csv                 # Test data
├── docs/
│   └── Report.md                         # Detailed experimental report
├── notebooks/
│   ├── data.ipynb                        # Exploratory data analysis
│   └── test.ipynb                        # Submission post-processing
├── outputs/
│   ├── predictions_roberta-large.csv
│   ├── predictions_microsoft_deberta-v3-large.csv
│   ├── predictions_deberta-v3-base.csv
│   ├── test_pred.csv
│   └── val_set_predictions_lr.csv
├── scripts/
│   ├── run_pipeline.sh                   # DeBERTa pipeline
│   └── run_roberta_pipeline.sh           # RoBERTa pipeline
├── check_id.py                            # Submission validator
├── download_model.py                      # Hugging Face model downloader
├── evaluate.py                            # Evaluation metrics
├── get_dist.py                            # Class distribution utility
├── predict.py                             # Test-set inference
├── predict_train_val.py                   # Train/validation predictions
├── run_baselines.py                       # Logistic Regression baseline
├── train.py                               # Transformer training
├── requirements.txt                       # Python dependencies
├── .env.example                           # Environment variable template
├── .gitignore
└── LICENSE
```

## Installation

Clone the repository and create a Python virtual environment:

```bash
git clone https://github.com/<your-username>/absa-nlp-contest.git
cd absa-nlp-contest

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Hugging Face Configuration

A Hugging Face token is optional. It can be used when downloading gated models or when higher Hugging Face Hub rate limits are required.

```bash
cp .env.example .env
```

Then add your token to `.env`:

```text
HF_TOKEN="your_token_here"
```

Do not commit `.env` or other secrets to the repository.

## Usage

### Run the Baseline

Run the Bag-of-Words and Logistic Regression baseline:

```bash
python run_baselines.py
```

### Train a Transformer Model

Fine-tune RoBERTa-large:

```bash
python train.py \
    --model_name roberta-large \
    --epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5
```

Fine-tune DeBERTa-v3-large with gradient accumulation:

```bash
python train.py \
    --model_name microsoft/deberta-v3-large \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4
```

### Generate Test Predictions

```bash
python predict.py \
    --model_dir ./trained_absa_model_roberta-large \
    --test_file data/contest2_test.csv \
    --output_file outputs/predictions_roberta-large.csv
```

### Validate a Submission

```bash
python check_id.py data/contest2_test.csv outputs/test_pred.csv
```

### Evaluate Predictions

```bash
python evaluate.py data/contest2_train.csv outputs/predictions_roberta-large.csv
```

### Run the Automated Pipelines

RoBERTa pipeline:

```bash
./scripts/run_roberta_pipeline.sh
```

DeBERTa pipeline:

```bash
./scripts/run_pipeline.sh
```

## Analysis

The Logistic Regression baseline uses Bag-of-Words features, which do not capture word order and contextual relationships effectively. This makes it difficult to distinguish mixed-sentiment reviews such as:

> The food was delicious, but the service was terrible.

The sequence-pair transformer approach substantially improves aspect detection, with both fine-tuned transformer models achieving Aspect Macro F1 scores above 0.92.

RoBERTa-large achieved a higher overall Micro F1 than DeBERTa-v3-base (0.751 vs. 0.735). Its larger model capacity provided an advantage in distinguishing more difficult sentiment classes, particularly minority classes such as `conflict` and `neutral`.

For detailed experiments, confusion matrices, and additional analysis, see [`docs/Report.md`](docs/Report.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.