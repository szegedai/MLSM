# Masked Latent Semantic Modeling (MLSM)

This is the source code accompanying the ACL Findings paper [_Masked Latent Semantic Modeling: an Efficient Pre-training Alternative to Masked Language Modeling_](https://aclanthology.org/2023.findings-acl.876/) and the follow-up [BabyLM system description paper](https://aclanthology.org/2023.conll-babylm.26/).

MLSM serves as a sample efficient alternative for traditional masked language modeling (MLM), during which the pre-training goal is not to recover the exact identity of the randomly selected subtokens, but to predict their distribution of latent semantic properties determined in an unsupervised and context-sensitive manner.

## Pre-trained model availability

A BERT medium model pre-trained using the HuggingFace library is also made available [via this link](https://huggingface.co/SzegedAI/bert-medium-mlsm).

MLSM was successfully used for pre-training language models for the [BabyLM Challange](https://babylm.github.io/) as our model was trained 2nd in the strict track when the amount of training data was limited to 10 million tokens.  
Our model submitted to the BabyLM challenge are also made available on HuggingFace Hub:
* BabyLM model trained on [10 million tokens](https://huggingface.co/SzegedAI/babylm-strict-small-mlsm)
* BabyLM model trained on [100 million tokens](https://huggingface.co/SzegedAI/babylm-strict-mlsm)

## Creating your own models

These are the steps you need to follow for training your own MLSM model.  

### Step 1 (Optional): Creating a custom tokenizer
This is an optional step. You might need to create a custom tokenizer on the data that you are about the pre-train your model on.  
If you are happy with an already existing tokenizer, this step can be safely ignored.

For this step, you can use the following command   

``` python train_tokenizer.py --vocab_size $VOCAB_SIZE --folder $INPUT_DATA_FOLDER --out_folder $TOKENIZER_NAME --cased```   

which will create a cased [Unigram](https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models) tokenizer consisting of ```$VOCAB_SIZE``` entries based on all the files in directory ```$INPUT_DATA_FOLDER```, saving the tokenizer into ```$TOKENIZER_NAME```.

### Step 2 (Optional): Creating an auxiliary model
MLSM relies on the hidden states of an existing pre-trained auxiliary model.  
One can use an already existing checkpoint, such as [```bert-base-cased```](https://huggingface.co/google-bert/bert-base-cased) in which case this step can be skipped as well.  

Our experimental results from our [BabyLM system description paper](https://aclanthology.org/2023.conll-babylm.26/) have shown that one does not necessarily need to rely on a fully pre-trained auxiliary model, but a partially pre-trained model created from scratch during a limited amount of update steps can also provide useful training signal for the MLSM pre training.

In case you wish to pre-train an auxiliary model with the standard MLM training objective, you can use the below command 

```
python pretrainer.py --transformer ${MODEL} \\
                     --reinit \\
                     --tokenizer ${TOKENIZER_NAME} \\
                     --out_dir ${AUXILIARY_MODEL_LOCATION} \\
                     --data_location ${PRETRAINING_DATA} \\
                     --training_seqs 102400000 \\
                     --batch 64 --grad_accum 16
```
As for defining the ```${MODEL}```, it is possible to provide a custom model config file, but it is also possible to refer to an already existing model configuration, such as ```google-bert/bert-base-cased```.  

Do not worry, in the latter case you are not going to initialize your auxiliary model with the weights of the referenced pre-trained model, but they will be reinitalized (as the ```--reinit``` flag also indicates it).  This means that it is only the model definition which are used from that model, the weights are going to be randomly initialized.  
If you wish not to reinitalize the base model, you can also change the second command line argument to ```--not-reinit```, in which case you are basically performing continual pre-training from an already pre-trained checkpoint without the random reinitalization of the model weights.

The ```${TOKENIZER_NAME}``` variable references to the tokenizer to be employed either in the form of a HuggingFace Hub repository name, or a path from the file system (such as the output folder name provided during Step 1).  

The pre-trained model will be saved into the folder ```${AUXILIARY_MODEL_LOCATION}```.  
There will be subfolders created according to the intermediate checkpoints created at 10%, 25%, 50% and 100% readiness level of the pre-training.

The ```$PRETRAINING_DATA``` refers to the location of the file(s) with the pre-training data.  
You can bundle all your pre-training data into a single file and provide its path for the given variable, or it is also possible to enter a regular expression for multiple files (such as ```/pre_training/data/location/*.gz```).  
The input files are expected to be (optionally gzipped) UTF-8 encoded text files one sentence per line (document boundaries indicated by empty lines).

The ```--training_seqs``` command line argument indicates how many input sequences to be considered during the pre-training.  
The effective batch size is the product of the ```--batch``` and ```---grad_accum``` command line arguments, standing for the batch size and the gradient accumulation, respectively.  
In the above example, where both command line arguments are set to 32, the number of effective batch size is ```64x16=1024```.  
As the ```--training_seqs``` is set to ```102400000``` in the example, it means that there are a total of ```102400000/(64x16)=100000``` update steps to be performed.

### Step 3: Learning the dictionary matrix for sparse coding

MLSM relies on performing sparse coding on the hidden states of an auxiliary model (which can either come from a pre-trained model from HuggingFace Hub or it can be pre-trained as described in Step 2).  

The dictionary matrix can be leared with the below command
```
python train_dict.py --transformer ${AUXILIARY_MODEL} \\
                     --corpus ${PRETRAINING_CORPUS} \\
                     --output ${SAVE_LOCATION} \\
                     --layer ${LAYER_TO_USE} \\
```
The ```$AUXILIARY_MODEL``` specifies a HuggingFace Hub model identifier or a path to a directory where auxiliary model to be used for obtaining the latent semantic knownledge to be found.  
(A separate tokenizer can also be specified, but it is not necessary if the $AUXILIARY_MODEL also comes with its own tokenizer.)  

The dictionary matrix is to be saved under ```$SAVE_LOCATION```.  

```$LAYER_TO_USE``` specifies the layer of the hidden representations of the auxiliary model to be used for dictionary learning.  
This value is best set to the last or the penultimate layer of the auxiliary model.  
For instance, for a 12 layer base-sized model, using the value 11 or 12 would be a reasonable choice.

The dictionary learning procedure has two hyperparameters, the number of semantic atoms and the regularization coefficient affecting the sparsity of the latent semantic representations to learn from.  

[We showed it earlier](https://aclanthology.org/2020.emnlp-main.683/) that the quality of the sparse representations obtained via dictionary learning is fairly robust to these choices.  
Setting the number of dictionary atoms to `3000` and the regularization coefficient to `0.05` is generally a reliable choice of these hyperparameters (which are also the default values emplyed by this script).

### Step 4: Pre-training using MLSM

Pre-training with MLSM can be performed by running the same script that can be used for pre-training an auxiliary model, but requires a few extra command line parameters to be set as illustrated by the code snippet below:

```
python pretrainer.py --transformer ${MLSM_MODEL} \\
                     --reinit \\
                     --tokenizer ${TOKENIZER_NAME} \\
                     --out_dir ${AUXILIARY_MODEL_LOCATION} \\
                     --data_location ${PRETRAINING_DATA} \\
                     --training_seqs 102400000 \\
                     --batch 64 --grad_accum 16 \\
                     --transformer2 ${AUXILIARY_MODEL} \\
                     --dict_file ${DICTIONARY_LOCATION} \\
                     --layer 11 \\
                     --mlm_weight 0.2
```
The role of the command line parameters are the same as for Step 2, the main differences are the following:  
* the `--transformer2` command line argument specifies the HuggingFace Hub identifier or the path of the auxiliary model to determine the target training distributions for pre-training,
* the `--dict_file` argument specifies the location of the dictionary matrix learned during Step 3,  
* the `--layer` argument has the same role as it had during the dictionary learning,
* the script also allows for training MLSM jointly with MLM pre-training: if you want to train towards both losses, the extent to which MLM weight is considered is controlled by the parameter provided for `--mlm_weight`. E.g., the `0.2` value specified in the example means that the final loss to be employed during the pre-training is going to be $`\mathcal{L}_{MLSM} + 0.2 * \mathcal{L}_{MLM}`$.  In cases, when the MLM capabilites of the pre-trained model are not important, this value can be safely set to 0 (without affecting the fine-tuning capabilities of the model).  This additional loss term becomes useful if the model is also expected to provide meaningful replacement of `[MASK]`ed tokens. In the BabyLM Challenge this was the case, and the MLM loss was weighted equally to the MLSM loss term (i.e., `--mlm_weight 1.0`). This hyperparameter was not optimized due to limitations of computational resources, but if you do not want to test your models capabilities on tasks that require traditional MLM capabilities, going for the default value (which is `0.0`) should also be sufficient.

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
```
@inproceedings{berend-2023-better,
    title = "Better Together: Jointly Using Masked Latent Semantic Modeling and Masked Language Modeling for Sample Efficient Pre-training",
    author = "Berend, G{\'a}bor",
    editor = "Warstadt, Alex  and
      Mueller, Aaron  and
      Choshen, Leshem  and
      Wilcox, Ethan  and
      Zhuang, Chengxu  and
      Ciro, Juan  and
      Mosquera, Rafael  and
      Paranjabe, Bhargavi  and
      Williams, Adina  and
      Linzen, Tal  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.conll-babylm.26",
    doi = "10.18653/v1/2023.conll-babylm.26",
    pages = "298--307",
}
```
