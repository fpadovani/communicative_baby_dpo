import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from utils_ppo import *


CSV_PATH = "./dpo_dataset/len_pairs_no_overlap_1.csv"
OUTPUT_JSON_REAL = "./dpo_dataset/huggingface_dpo_format.json"
OUTPUT_JSON_SYNTHETIC = "./dpo_dataset/synthetic_dpo_format.json"
HF_HUB_NAME = "fpadovani/child-dpo-preferences"
MODEL_NAME = "meta-llama/Llama-3.2-3B"
BATCH_SIZE = 16

def split_turns(pair):
    try:
        mot, chi = pair.split("\t")
        return mot.strip(), chi.strip()
    except:
        return None, None



def create_dataset_real():
    df = pd.read_csv(CSV_PATH)
    df_train = df[:18000]

    preference_data = []
    for _, row in df_train.iterrows():
        pos_mot, pos_chi = split_turns(row["pospair"])
        neg_mot, neg_chi = split_turns(row["negpair"])
        if pos_mot and neg_mot and pos_mot == neg_mot:
            preference_data.append({
                "prompt": pos_mot,
                "chosen": pos_chi,
                "rejected": neg_chi
            })

    with open(OUTPUT_JSON_REAL, "w", encoding="utf-8") as f:
        json.dump(preference_data, f, indent=2, ensure_ascii=False)

    dataset = Dataset.from_list(preference_data)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.push_to_hub(HF_HUB_NAME)

    print(f"✅ Real dataset saved to {OUTPUT_JSON_REAL} and pushed to HuggingFace.")



def generate_teacher_responses_vllm_batch(llm, mother_sents, sampling_params):
    system_msg = (
        "You are a young child having a conversation with your mother. "
        "When your mother says something, you should answer as a typical kind and natural-sounding child. "
        "Do NOT repeat her words. Instead, give a new, relevant answer that shows understanding. "
        "Keep it short and child-like."
    )
    full_prompts = [f"{system_msg}\nMother says: {mother_sents[i]}\nChild:" for i in range(len(mother_sents))]
    outputs = llm.generate(full_prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def create_dataset_synthetic():
    df = pd.read_csv(CSV_PATH)
    df_train = df[:18000]

    prompts = []
    neg_pair =[]

    for _, row in df_train.iterrows():
        mot, chi = split_turns(row["pospair"])
        if mot and chi:
            prompts.append(mot.replace("*MOT:", "").strip())
    
    for _, row in df_train.iterrows():
        mot, chi = split_turns(row["negpair"])
        if mot and chi:
            neg_pair.append(chi)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=30,  # can be adjusted dynamically based on chi_lens if needed
        stop=["\n", "Mother:", "Child:"],
        skip_special_tokens=True,
    )

    llm = LLM(
        model=MODEL_NAME,
        dtype="float16",
        gpu_memory_utilization=0.2,
        max_model_len=512,
        trust_remote_code=True,
        enforce_eager=True,
        disable_log_stats=True,
        tensor_parallel_size=1,
    )

    preference_data = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i:i+BATCH_SIZE]
        neg_pairs_batch = neg_pair[i:i+BATCH_SIZE]

        responses = generate_teacher_responses_vllm_batch(llm, batch_prompts, sampling_params)
        teacher_utts = [extract_first_utterance(t) for t in responses]

        for mot, chi_response, neg_response in zip(batch_prompts, teacher_utts, neg_pairs_batch):
            preference_data.append({
                "prompt": "*MOT: " + mot,
                "chosen": "*CHI: " + chi_response.lower(),
                "rejected": neg_response  # No negative samples in this case
            })

    with open(OUTPUT_JSON_SYNTHETIC, "w", encoding="utf-8") as f:
        json.dump(preference_data, f, indent=2, ensure_ascii=False)

    dataset = Dataset.from_list(preference_data)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.push_to_hub(HF_HUB_NAME + "-synthetic")

    print(f"Synthetic dataset saved to {OUTPUT_JSON_SYNTHETIC} and pushed to HuggingFace.")


if __name__ == "__main__":
    print("Choose dataset generation mode:")
    print("1. Use real child utterances (pospair/negpair)")
    print("2. Use LLM to generate child utterances")

    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        create_dataset_real()
    elif choice == "2":
        create_dataset_synthetic()
    else:
        print("Invalid input. Exiting.")