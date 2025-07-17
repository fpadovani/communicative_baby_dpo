# DPO Training to finetune a Baby and make it more Communicative

## Model

As baseline, we use the model pre-trained by Bastian -> [Baseline_baby](https://huggingface.co/bbunzeck/another-llama)

## DPO Dataset

Bastian has taken the two data splits (comprising communicative turns - triplets - between a MOT/FAT/INV and a CHI) that he didn't use for training, \
specifically childes-dialogue2.txt and childes-dialogue3.txt, and he extracted real minimal pair interactions for instance: \
** *MOT: what is that ? *CHI: it looks like a gun .**
