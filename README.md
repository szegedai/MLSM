# Masked Latent Semantic Modeling (MLSM)

This is the source code for the paper [_Masked Latent Semantic Modeling: an Efficient Pre-training Alternative to Masked Language Modeling_](https://aclanthology.org/2023.findings-acl.876/).

MLSM serves as a sample efficient alternative for traditional masked language modeling (MLM), during which the pre-training goal is not to recover he exact identity of the randomly selected subtokens, but to predict their distribution of latent semantic properties determined in an unsupervised and context-sensitive manner.

## Pre-trained model availability

A BERT medium model pre-trained using the HuggingFace library is also made available [via this link](https://huggingface.co/SzegedAI/bert-medium-mlsm).

## BibTeX

```
@inproceedings{berend-2023-masked,
    title = "Masked Latent Semantic Modeling: an Efficient Pre-training Alternative to Masked Language Modeling",
    author = "Berend, G{\'a}bor",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.876",
    pages = "13949--13962",
    abstract = "In this paper, we propose an alternative to the classic masked language modeling (MLM) pre-training paradigm, where the objective is altered from the reconstruction of the exact identity of randomly selected masked subwords to the prediction of their latent semantic properties. We coin the proposed pre-training technique masked latent semantic modeling (MLSM for short). In order to make the contextualized determination of the latent semantic properties of the masked subwords possible, we rely on an unsupervised technique which uses sparse coding. Our experimental results reveal that the fine-tuned performance of those models that we pre-trained via MLSM is consistently and significantly better compared to the use of vanilla MLM pretraining and other strong baselines.",
}
```
