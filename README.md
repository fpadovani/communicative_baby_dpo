# DPO Training to finetune a Baby and make it more Communicative

## Model

As baseline, we use the model pre-trained by Bastian -> [Baseline_baby](https://huggingface.co/bbunzeck/another-llama)

## DPO Dataset

Bastian has taken the two data splits (comprising communicative turns - triplets - between a MOT/FAT/INV and a CHI) that he didn't use for training, \
specifically childes-dialogue2.txt and childes-dialogue3.txt, and he extracted real minimal pair interactions for instance: \
<pre><code> *MOT: what is that ? *CHI: it looks like a gun .</code></pre>

From these he generated 4 .txt minimal pairs files: \

- mother question + correct child answer vs. mother question + incorrect child answer with matched length *(n_words)*

- mother question + correct child answer vs. mother question + incorrect child answer with matched length *(n_subword tokens)*
