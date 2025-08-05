# DPO Training to finetune a Baby and make it more Communicative

To run correctly the code in this repository you need the last version of the trl library. 

## Model

As baseline, we use the model pre-trained by Bastian -> [Baseline_baby](https://huggingface.co/bbunzeck/another-llama)

## DPO Dataset

Bastian has taken the two data splits (comprising communicative turns - triplets - between a MOT/FAT/INV and a CHI) that he didn't use for training,
specifically childes-dialogue2.txt and childes-dialogue3.txt, and he extracted real minimal pair interactions that involve not only questions, but all kinds of MOT-CHI tuples, such as: 
<pre><code> *MOT: what is that ? *CHI: it looks like a gun .</code></pre>

From these he generated 4 .txt minimal pairs files: 

- mother question + correct child answer vs. mother question + incorrect child answer with matched length **(n_words)** (with or without overlap between MOT and CHI utterances)

- mother question + correct child answer vs. mother question + incorrect child answer with matched length **(n_subword tokens)** (with or without overlap between MOT and CHI utterances)


These files can be found in the `./dpo_dataset` folder:

1. `len_pairs_no_overlap_1.csv` -> total of 25547 min pairs
2. `len_pairs_overlap_1.csv` -> total of 88171 min pairs
3. `tok_pairs_no_overlap_1.csv` -> total of 25519 min pairs
4. `tok_pairs_overlap_1.csv` -> total of 88136 min pairs


I have taken the first file to generate a DPO training dataset split and an evaluation dataset split to be used to evaluate baseline and fine-tuned models (as we agreed before), these are based on matched amount of words (tokens).
I used 18000 rows for the training and the rest for evaluation.

I created two types of training data:

- the first one uses realistic minimal pairs (as they occur in the .txt files and as they are extracted from CHILDES by Bastian) -> [**dpo_dataset/huggingface_dpo_format.json**](https://huggingface.co/datasets/fpadovani/child-dpo-preferences)
- the second one instead takes the *MOT: sentence as a prompt to a LLM (Teacher) that tries to simulate a good *CHI: answer -> [**dpo_dataset/synthetic_dpo_format.json**](https://huggingface.co/datasets/fpadovani/child-dpo-preferences-synthetic)


This is the prompt I used, it can be found in the `generate_dpo_pref_from_file.py` file.


<pre><code>
"You are a young child having a conversation with your mother. "
"When your mother says something, you should answer as a typical kind and natural-sounding child. "
"Do NOT repeat her words. Instead, give a new, relevant answer that shows understanding. "
"Keep it short and child-like."
</code></pre>



- this is the dataset split for evaluation -> [**dpo_dataset/huggingface_dpo_format_eval.json**](https://huggingface.co/datasets/fpadovani/dialogue_eval_tokens) \

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

**BASELINE**: our *bbunzeck/another-llama* baseline model scores **65.5%** (accuracy) on Zorro and **64.3%** on the minimal pairs evaluation set based on words match, and **63.8%** on dialogue minimal pairs based on tokens match. It scores **40.3%** on the lexical decision task. \

**DPO_REAL_PAIRS**: the last checkpoint of our fine-tuned model on real dpo pairs scores **64.8%** on Zorro and **68.4%** on the minimal pairs evaluation set based on words match, and **67.6%** on dialogue minimal pairs based on tokens match. It scores **40.5%** on the lexical decision task. \

**DPO_SYNTHETIC_PAIRS**: the last checkpoint of our fine-tuned model on real dpo pairs scores **62.7%** on Zorro and **64.9%** on the minimal pairs evaluation set based on words match, and **64.3%** on dialogue minimal pairs based on tokens match. It scores **41.3%** on the lexical decision task.\

**RESULTs**: I wouldn't call degradation in performance the 1 % decrease in Zorro accuracy, it can be considered noise. Good that we have an improvement of accuracy (even if small) on dialogue minimal pairs after fine-tuning. \


## Plots of reward and loss 
In the `./plots` folder you can find the loss trend and the reward trend for the correct and incorrect sentences. 
The curves make a lot of sense and for the fine-tuning with synthetic dataset they look even more stable. 


