# DPO Training to finetune a Baby and make it more Communicative

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


I have taken the first file to generate a DPO training dataset and and an evaluation dataset to be used to test baseline and fine-tuned models.
I used 18000 rows for the training and the rest for evaluation.

I created two types of training data:

- the first one uses realistic minimal pairs (as they occur in the .txt files and as they are extracted from CHILDES by Bastian) -> [**dpo_dataset/huggingface_dpo_format.json**](https://huggingface.co/datasets/fpadovani/child-dpo-preferences)
- the second one instead takes the *MOT: sentence as a prompt to a LLM (Teacher) that tries to simulate a good *CHI: answer -> [**dpo_dataset/synthetic_dpo_format.json**](https://huggingface.co/datasets/fpadovani/child-dpo-preferences-synthetic)


This is the prompt I used, it can be found in the `generate_dpo_pref_from_file.py` file.


<pre><code>"You are a young child having a conversation with your mother. "
        "When your mother says something, you should answer as a typical kind and natural-sounding child. "
        "Do NOT repeat her words. Instead, give a new, relevant answer that shows understanding. "
        "Keep it short and child-like." </code></pre>


*PS: as you can see, I made a mistake while generating the synthetic dataset, since the tags *MOT: and *CHI are missing. I already adjusted the code in the `generate_dpo_pref_from_file.py` file, and we only need to rerun the data generation and the DPO training with the correct file.*


- this is the dataset split for evaluation -> [**dpo_dataset/huggingface_dpo_format_eval.json**](https://huggingface.co/datasets/fpadovani/child-dpo-preferences-eval)

