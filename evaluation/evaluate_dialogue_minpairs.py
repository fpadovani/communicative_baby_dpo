import pandas as pd
from datasets import load_dataset
from minicons import scorer
import torch
from tqdm import tqdm

# CONFIG: Choose your model and dataset
DIALOGUE_WORDS= "fpadovani/dialogue_eval_words" 
DIALOGUE_TOKENS = "fpadovani/dialogue_eval_tokens" 
INTERACTIVE = 'BabyLM-community/babylm-interaction-baseline-simpo'
GPT_BERT = 'BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus'
BASELINE_PATH = "bbunzeck/another-llama"
FINETUNED_PATH_1 = "./finetuned_models/communicative-baby-dpo-synthetic/checkpoint-5630"  # Local path to first fine-tuned model
FINETUNED_PATH_2 = "./finetuned_models/communicative-baby-dpo/checkpoint-5630"
SPLIT = "train"

# Load dataset from HuggingFace
print(f"🔄 Loading HuggingFace evaluation datasets words: {DIALOGUE_WORDS}")
dataset_words = load_dataset(DIALOGUE_WORDS, split=SPLIT)
print(f"✅ Loaded {len(dataset_words)} samples.")

# Load dataset 2 from HuggingFace
print(f"🔄 Loading HuggingFace evaluation datasets tokens: {DIALOGUE_TOKENS}")
dataset_tokens = load_dataset(DIALOGUE_TOKENS, split=SPLIT)
print(f"✅ Loaded {len(dataset_tokens)} samples.")



# Convert to list of dicts for iteration
data_words = dataset_words.to_list()
data_tokens = dataset_tokens.to_list()

# Load MiniCONS models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
interactive = scorer.IncrementalLMScorer(INTERACTIVE, device=device, trust_remote_code=True)
gpt_bert = scorer.IncrementalLMScorer(GPT_BERT, device=device, trust_remote_code=True)
baseline_model = scorer.IncrementalLMScorer(BASELINE_PATH, device=device)
finetuned_model_1 = scorer.IncrementalLMScorer(FINETUNED_PATH_1, device=device)
finetuned_model_2 = scorer.IncrementalLMScorer(FINETUNED_PATH_2, device=device)

# Evaluation function
def evaluate_model(model, data):
    correct = 0
    total = len(data)

    for row in tqdm(data):
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        pos_input = prompt + " " + chosen
        neg_input = prompt + " " + rejected

        pos_score = model.sequence_score(pos_input, reduction = lambda x: x.sum(0).item(), bow_correction=True)
        neg_score = model.sequence_score(neg_input, reduction = lambda x: x.sum(0).item(), bow_correction=True)

        if pos_score > neg_score:
            correct += 1

    return correct / total


interactive_words = evaluate_model(interactive, data_words)
print(f"Interactive model accuracy: {interactive_words:.3f}")

gpt_bert_words = evaluate_model(gpt_bert, data_words)
print(f"Gpt-BErt model accuracy: {gpt_bert_words:.3f}")

baseline_words = evaluate_model(baseline_model, data_words)
print(f"Baseline model accuracy: {baseline_words:.3f}")

finetuned_1_words = evaluate_model(finetuned_model_1, data_words)
print(f"Fine-tuned model accuracy: {finetuned_1_words:.3f}")

finetuned_2_words = evaluate_model(finetuned_model_2, data_words)
print(f"Fine-tuned model accuracy: {finetuned_2_words:.3f}")

interactive_tokens = evaluate_model(interactive, data_tokens)
print(f"Interactive model accuracy: {interactive_tokens:.3f}")

gpt_bert_tokens = evaluate_model(gpt_bert, data_tokens)
print(f"Gpt-BErt model accuracy: {gpt_bert_tokens:.3f}")

baseline_tokens = evaluate_model(baseline_model, data_tokens)
print(f"Baseline model accuracy: {baseline_tokens:.3f}")

finetuned_1_tokens = evaluate_model(finetuned_model_1, data_tokens)
print(f"Fine-tuned model accuracy: {finetuned_1_tokens:.3f}")

finetuned_2_tokens = evaluate_model(finetuned_model_2, data_tokens)
print(f"Fine-tuned model accuracy: {finetuned_2_tokens:.3f}")

