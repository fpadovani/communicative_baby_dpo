from datasets import load_dataset

# Load datasets
dataset1 = load_dataset("fpadovani/child-dpo-preferences-synthetic", split="train")
dataset2 = load_dataset("fpadovani/child-dpo-preferences", split="train")

def count_whitespace_tokens(dataset):
    total_tokens = 0
    for row in dataset:
        text = (row["prompt"] or "") + " " + (row["chosen"] or "")
        total_tokens += len(text.strip().split())
    return total_tokens

tokens_dataset1 = count_whitespace_tokens(dataset1)
tokens_dataset2 = count_whitespace_tokens(dataset2)

print(f"Dataset 1 total whitespace tokens: {tokens_dataset1:,}")
print(f"Dataset 2 total whitespace tokens: {tokens_dataset2:,}")