# BabyLM Challenge 2025 - DPO fine-tuning for a Baby Model 

To run correctly the code in this repository you need the last version of the trl library. 

## Model

As baseline, we use the a model pre-trained on dialogue turns between a child and a caregiver  -> [Llamalogue](https://huggingface.co/CLAUSE-Bielefeld/llamalogue/tree/main)

## DPO Datasets to fine-tune the model

1. the first one uses realistic minimal pairs (as they occur in the our pre-processed triplets files used to train Llamalogue and as they are extracted from CHILDES transcripts) -> [**dpo_dataset/huggingface_dpo_format.json**](https://huggingface.co/datasets/fpadovani/child-dpo-preferences)
2. the second one instead features rows that show a mother’s utterance (MOT) as a real prompt, with corresponding appropriate child responses (CHI) generated using the Llama-3.2-3B teacher model, and random responses from naturalistic mismatches (as the previous one) -> [**dpo_dataset/synthetic_dpo_format.json**](https://huggingface.co/datasets/fpadovani/child-dpo-preferences-synthetic)


This is the prompt used, it can be found in the `generate_dpo_pref_from_file.py` file.


<pre><code>
"You are a young child having a conversation with your mother. "
"When your mother says something, you should answer as a typical and natural-sounding child. "
"Do NOT repeat her words. Instead, give a new, relevant answer that shows understanding. "
"Keep it short and child-like."
</code></pre>



- this is the dataset split for evaluation with appropriate and random sentence matched in terms of word length -> [**dpo_dataset/huggingface_dpo_format_eval.json**](https://huggingface.co/datasets/fpadovani/dialogue_eval_tokens) \

Moreover, I also used the `tok_pairs_no_overlap_1.csv` file to generate another evaluation set of minimal pairs based on matched amount of tokens between the pairs. The dataset split in this case is the following -> [**dpo_dataset/huggingface_dpo_format_eval_tokens.json**](https://huggingface.co/datasets/fpadovani/dialogue_eval_tokens)
  

## Training with DPO
Using the `dpo_training.py` script, changing the dataset in input (either the huggingface_dpo_format.json or synthetic_dpo_format.json) I fine-tuned for 10 epochs the baseline model, saving checkpoints every 2000 steps. 

The fine-tuned models can be found here:
- [fpadovani/communicative-baby-dpo](https://huggingface.co/fpadovani/communicative-baby-dpo)
- [fpadovani/communicative-baby-dpo-synthetic](https://huggingface.co/fpadovani/communicative-baby-dpo-synthetic)

## Evaluation with DPO
We should familiarize with the BabyLM Challenge evaluation pipeline of this year -> [2025](https://github.com/babylm/evaluation-pipeline-2025)

In the meantime I have scripts that evaluate our baseline and finetuned models on Zorro, on our own minimal dialogue pair dataset (with words matched length and token matched length) and on single lexical items (taken from Bastian lexical decision task paper):

- *`./evaluation/evaluate_zorro.py`* 
- *`./evaluation/evaluate_dialogue_minpairs.py`*
- *`./evaluation/evaluate_lexicon.py`*

**BASELINE**: the baseline model provided by the organizer of the BabyLM Challenge, it is trained on 10M tokens, but winning architecture from last year. [BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus][https://huggingface.co/BabyLM-community/babylm-baseline-10m-gpt-bert-causal-focus]. It scores **77.7%** (accuracy) on Zorro and **58.3%** on the dialogue minimal pairs evaluation set based on words match, and **57.4%** on dialogue minimal pairs based on tokens match. 
The result on lexical decision task is **56.7%**.

**PRE-TRAINED**: our *bbunzeck/another-llama* baseline model scores **65.5%** (accuracy) on Zorro and **64.3%** on the minimal pairs evaluation set based on words match, and **63.8%** on dialogue minimal pairs based on tokens match. It scores **40.3%** on the lexical decision task. \

**DPO_REAL_PAIRS**: the last checkpoint of our fine-tuned model on real dpo pairs scores **64.8%** on Zorro and **68.4%** on the minimal pairs evaluation set based on words match, and **67.6%** on dialogue minimal pairs based on tokens match. It scores **40.5%** on the lexical decision task. \

**DPO_SYNTHETIC_PAIRS**: the last checkpoint of our fine-tuned model on real dpo pairs scores **62.7%** on Zorro and **64.9%** on the minimal pairs evaluation set based on words match, and **64.3%** on dialogue minimal pairs based on tokens match. It scores **41.3%** on the lexical decision task.\


## Plots of reward and loss 
In the `./plots` folder you can find the loss trend and the reward trend for the correct and incorrect sentences. 
The curves make a lot of sense and for the fine-tuning with synthetic dataset they look even more stable. 


