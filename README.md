# BabyLM Challenge 2025 
### *DPO fine-tuning for a Communicative Baby Model*

To run correctly the code in this repository you need the last version of the trl library. 

## Model

As baseline, we use the a model pre-trained on dialogue turns between a child and a caregiver  -> [Llamalogue](https://huggingface.co/CLAUSE-Bielefeld/llamalogue/tree/main)

## DPO Datasets for Training 

1. the first one uses realistic minimal pairs (as they occur in the our pre-processed triplets files used to train Llamalogue and as they are extracted from CHILDES transcripts) -> [**dpo_dataset/huggingface_dpo_format.json**](https://huggingface.co/datasets/fpadovani/child-dpo-preferences)
2. the second one instead features rows that show a mother’s utterance (MOT) as a real prompt, with corresponding appropriate child responses (CHI) generated using the Llama-3.2-3B teacher model, and random responses from naturalistic mismatches (as the previous one) -> [**dpo_dataset/synthetic_dpo_format.json**](https://huggingface.co/datasets/fpadovani/child-dpo-preferences-synthetic)


This is the prompt used, it can be found in the `generate_dpo_pref_from_file.py` file.


<pre><code>
"You are a young child having a conversation with your mother. "
"When your mother says something, you should answer as a typical and natural-sounding child. "
"Do NOT repeat her words. Instead, give a new, relevant answer that shows understanding. "
"Keep it short and child-like."
</code></pre>

## DPO Datasets for Evaluation 

- this is the dataset split for evaluation with appropriate and random sentence matched in terms of word length -> [**dpo_dataset/huggingface_dpo_format_eval.json**](https://huggingface.co/datasets/fpadovani/dialogue_eval_words) \
- this is the dataset split for evaluation with appropriate and random sentence matched in terms of token length -> [**dpo_dataset/huggingface_dpo_format_eval_tokens.json**](https://huggingface.co/datasets/fpadovani/dialogue_eval_tokens) \

  

## Training with DPO
Using the `dpo_training.py` script, changing the dataset in input (either the huggingface_dpo_format.json or synthetic_dpo_format.json) we fine-tuned for 10 epochs the baseline model, saving checkpoints every 2000 steps. 

The fine-tuned models can be found here:
- [CLAUSE-Bielefeld/communicative-baby-dpo](https://huggingface.co/CLAUSE-Bielefeld/communicative-baby-dpo)
- [CLAUSE-Bielefeld/communicative-baby-dpo-synthetic](https://huggingface.co/CLAUSE-Bielefeld/communicative-baby-dpo-synthetic)

## Evaluation with DPO

Scripts that evaluate our baseline and finetuned models on Zorro, on our own minimal dialogue pair dataset (with words matched length and token matched length) and on single lexical items:

- *`./evaluation/evaluate_zorro.py`* 
- *`./evaluation/evaluate_dialogue_minpairs.py`*
- *`./evaluation/evaluate_lexicon.py`*

## Plots of reward and loss 
In the `./plots` folder you can find the loss trend and the reward trend for the correct and incorrect sentences. 
The curves make a lot of sense and for the fine-tuning with synthetic dataset they look even more stable. 


