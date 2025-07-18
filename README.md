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


I have taken the first file to generate a DPO training dataset split and an evaluation dataset split to be used to evaluate baseline and fine-tuned models (as we agreed before).
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



- this is the dataset split for evaluation -> [**dpo_dataset/huggingface_dpo_format_eval.json**](https://huggingface.co/datasets/fpadovani/child-dpo-preferences-eval)
  

## Training with DPO
Using the `dpo_training.py` script, changing the dataset in input (either the huggingface_dpo_format.json or synthetic_dpo_format.json) I fine-tuned for 10 epochs the baseline model, saving checkpoints every 2000 steps. 

The fine-tuned models can be found here:
- [fpadovani/communicative-baby-dpo](https://huggingface.co/fpadovani/communicative-baby-dpo)
- [fpadovani/communicative-baby-dpo-synthetic](https://huggingface.co/fpadovani/communicative-baby-dpo-synthetic)

## Evaluation with DPO
We should familiarize with the BabyLM Challenge evaluation pipeline of this year -> [2025](https://github.com/babylm/evaluation-pipeline-2025)

In the meantime I have a script that evaluate our baseline and finetuned models on BLIMP and on our own minimal dialogue pair dataset:

- *`evaluate_blimp.py`* 
- *`evaluate_minpairs.py`*

**BASELINE**: our *bbunzeck/another-llama* baseline model scores 56% (accuracy) on BLIMP and 64.4% on the minimal pairs evaluation set \

**DPO_REAL_PAIRS**: the last checkpoint of our fine-tuned model on real dpo pairs scores 55% on BLIMP and 68% on the minimal pairs evaluation set 

**DPO_SYNTHETIC_PAIRS**: the last checkpoint of our fine-tuned model on real dpo pairs scores 54% on BLIMP and 66.8% on the minimal pairs evaluation set


**RESULTs**: I wouldn't call degradation in performance the 0.1/0.2 % decrease in BLIMP accuracy, it can be considered noise. Good that we have an improvement of accuracy (even if small) on dialogue minimal pairs after fine-tuning. 


## Plots of reward and loss 
In the `./plots` folder you can find the loss trend and the reward trend for the correct and incorrect sentences. 
The curves make a lot of sense and for the fine-tuning with synthetic dataset they look even more stable. 


