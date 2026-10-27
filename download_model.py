import argparse
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load environment variables if .env exists
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Download pre-trained model and tokenizer from Hugging Face Hub")
    parser.add_argument('--model_name', type=str, default='roberta-large', help='Model identifier on HuggingFace Hub')
    parser.add_argument('--output_dir', type=str, default=None, help='Target directory to save model (defaults to ./models/<model_name>)')
    parser.add_argument('--num_labels', type=int, default=5, help='Number of classification labels')
    args = parser.parse_args()

    model_name = args.model_name
    sanitized_name = model_name.replace('/', '_')
    target_dir = args.output_dir or f"./models/{sanitized_name}"

    hf_token = os.getenv("HF_TOKEN")
    token_arg = hf_token if hf_token else None

    print(f"Downloading model '{model_name}' to '{target_dir}'...")
    os.makedirs(target_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token_arg, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=args.num_labels,
        token=token_arg
    )

    tokenizer.save_pretrained(target_dir)
    model.save_pretrained(target_dir)

    print(f"Successfully downloaded and saved '{model_name}' to: {target_dir}")

if __name__ == "__main__":
    main()
