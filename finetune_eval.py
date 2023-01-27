import os, sys
import argparse
import torch
import numpy as np
import random

import evaluate
from loader_helper import load_custom_dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMultipleChoice, AutoModelForTokenClassification, get_scheduler
from tqdm.auto import tqdm

from collections import Counter
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    stream=sys.stdout)

def get_optimizer_and_scheduler(m, training_steps, lr=2e-5):
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    #optimizer = torch.optim.SGD(m.parameters(), lr=lr)

    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=training_steps)
    return optimizer, scheduler


def pad_batch(batch, device):
    b = {}

    multiple_choice_task = len(batch['input_ids'][0].shape) > 1
    token_classification_task = type(batch['labels'])==list

    if multiple_choice_task == True:
        max_length = max([len(choice) for b in batch['input_ids'] for choice in b])
    else:
        max_length = max([len(b) for b in batch['input_ids']])

    for f in batch.features:
        if multiple_choice_task:
            if f=='labels':
                b[f] = batch[f].to(device)
            else:
                batch_vals = []
                for batch_item in batch[f]:
                    choices = []
                    for choice in batch_item:
                        choices.append(torch.cat((choice, torch.zeros(max_length - len(choice), dtype=choice.dtype))))
                    batch_vals.append(torch.vstack(choices))

                b[f] = torch.stack(batch_vals).to(device)
        else:
            if f=='labels' and token_classification_task==False:
                b[f] = batch[f].to(device)
            else:
                if f=='labels': # for token classification labels
                    b[f] = torch.vstack([torch.cat((b, -100*torch.ones(max_length - len(b), dtype=b.dtype))) for b in batch[f]]).to(device)
                else:
                    b[f] = torch.vstack([torch.cat((b, torch.zeros(max_length - len(b), dtype=b.dtype))) for b in batch[f]]).to(device)
    return b

def train(model, metric, num_epochs, training_data, eval_data, tokenizer,
          batch_size, grad_accum, lr=2e-5, eval_data_ids=None, shuffle_seed=42):
 
    metrics, losses = [[] for _ in range(len(eval_data))], [0.0]
    
    num_training_steps = num_epochs * int(np.ceil(len(training_data) / (batch_size * grad_accum)))
    num_eval_steps = num_epochs * (0 if eval_data is None else sum([int(np.ceil(len(ed) / batch_size)) for ed in eval_data]))

    full_macrobatches_per_epoch = len(training_data) // (batch_size * grad_accum)
    minibatches_per_epoch = int(np.ceil(len(training_data) / batch_size))
    last_minibatch_number = int(np.ceil((len(training_data) - full_macrobatches_per_epoch * (batch_size * grad_accum)) / batch_size))
    
    optimizer, lr_scheduler = get_optimizer_and_scheduler(model, num_training_steps, lr)
    progress_bar = tqdm(range(num_training_steps + num_eval_steps), file=sys.stderr)

    np.random.seed(shuffle_seed)
    
    for epoch in range(num_epochs):
        model.train()

        random_samples = np.random.permutation(range(len(training_data)))
        for bi, from_idx in enumerate(range(0, len(training_data), batch_size)):
            to_index = min(len(training_data), from_idx+batch_size)
            batch = pad_batch(training_data.select(random_samples[from_idx:to_index]), model.device)

            if epoch==0 and bi==0:
                if len(batch['input_ids'].shape)==2:
                    logging.info(batch['input_ids'][0:3,0:8])
                else: # multiple choice task
                    logging.info(batch['input_ids'][0:3,:,0:8])
            outputs = model(**batch)
    
            normalizer = grad_accum if bi < grad_accum * full_macrobatches_per_epoch else last_minibatch_number

            loss = outputs.loss / normalizer
            losses[-1] += loss.item()
            loss.backward()

            if ((bi+1) % grad_accum == 0) or bi == minibatches_per_epoch - 1:
                optimizer.step() 
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                losses.append(0.0)

        if metric is not None and eval_data is not None:
            for ed_id, ed in enumerate(eval_data):
                performance, aggregated_preds = eval_model(model, ed, batch_size, metric, progress_bar, eval_data_ids[ed_id], tokenizer)
                metrics[ed_id].append(performance)
                if len(aggregated_preds) > 0 and type(aggregated_preds[0]) != list:
                    logging.info(("Predictions@{}:".format(epoch), [pred.item() for pred in aggregated_preds]))

    return metrics, losses


def eval_model(model, eval_data, batch_size, metric, progress_bar, eval_data_id, tokenizer):
    def map_seq_id(labels_list, label):
        if label==0:
            labels_list.append('O')
        elif label % 2 == 1:
            labels_list.append('B-{}'.format(label.item() // 2))
        else:
            labels_list.append('I-{}'.format((label.item()-1) // 2))
            
    model.eval()
    aggregated_preds, preds, golds = [], [], []
    for bi in range(int(np.ceil(len(eval_data) / batch_size))):

        to_index = min(len(eval_data), (bi+1)*batch_size)
        batch = pad_batch(eval_data.select(range(bi*batch_size, to_index)), model.device)

        with torch.no_grad():
            outputs = model(**batch)

        if len(outputs.logits.shape)==2 and outputs.logits.shape[1]==1: # regression
            preds, golds = outputs.logits, batch['labels']
        else:
            predictions = torch.argmax(outputs.logits, dim=-1)
            if eval_data_id == '-hans':
                predictions[predictions==2] = 1 # in HANS, the neutral and contradiction labels are merged
            preds, golds = predictions, batch['labels']
            if len(preds.shape)==2: # this implies token classification
                preds, golds = [], []
                for i in range(predictions.shape[0]):
                    local_preds, local_golds = [], []
                    for pred_label, gold_label in zip(predictions[i], batch['labels'][i]):
                        if gold_label!=-100:
                            map_seq_id(local_preds, pred_label)
                            map_seq_id(local_golds, gold_label)
                preds.append(local_preds)
                golds.append(local_golds)

        aggregated_preds.extend(preds)
        metric.add_batch(predictions=preds, references=golds)
        if progress_bar is not None:
            progress_bar.update(1)
    return metric.compute(), aggregated_preds


def get_dataloaders(dataset_name, tokenizer, subcorpus, label_name='labels', samples_to_use=-1):

    def tokenize(examples):
        return tokenizer(examples['sentence'], padding=True, truncation=True)

    def sentence_pair_tokenize(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], padding=True, truncation=True)

    def multiple_choice_tokenize(examples):
        first_sentences = [context for context in examples["premise"] for _ in range(num_choices)]
        second_sentences = [choice for choices in zip(*[examples['choice{}'.format(i+1)] for i in range(num_choices)]) for choice in choices]
        tokenized_examples = tokenizer(first_sentences, second_sentences, padding=True, truncation=True)
        return {k: [v[i : i + num_choices] for i in range(0, len(v), num_choices)] for k, v in tokenized_examples.items()}

    def seq_tokenize(examples):
        inputs = tokenizer(examples['tokens'], padding=True, is_split_into_words=True, truncation=True, return_offsets_mapping=True, return_length=True)
        batch_labels = []
        for offsets, tags, att, length, toks in zip(inputs['offset_mapping'], examples[label_name], inputs['attention_mask'], inputs['length'], examples['tokens']):
            labels = [-100]
            position = 0
            l = sum(att)
            for i,offset in enumerate(offsets[1:l-1]):
                next_label = -100
                if offset[0]==0: # this is a non-continued subtoken
                    next_label = tags[position]
                    position += 1
                labels.append(next_label)
            labels.extend([-100 for _ in range(l-1, length)])
            batch_labels.append(labels)
        inputs['labels'] = batch_labels
        return inputs

    if dataset_name.startswith('NYTK/Hu') or 'opinhubank' in dataset_name:
        data = load_custom_dataset(dataset_name, subcorpus, delete_json=True)
    else:
        data = load_dataset(*dataset_name.split('-'))[subcorpus]

    if dataset_name == 'imdb':
        data = data.rename_column('text', 'sentence').rename_column('label', label_name).map(tokenize, batched=True)
    elif 'opinhubank' in dataset_name:
        data = data.map(sentence_pair_tokenize, batched=True)

    elif dataset_name.startswith('NYTK/Hu'):
        if 'WNLI' in dataset_name:
            data = data.map(sentence_pair_tokenize, batched=True)
        elif 'CoPA' in dataset_name:
            num_choices = sum([1 for f in data.features if f.startswith('choice')])
            data = data.map(multiple_choice_tokenize, batched=True)
        else:
            data = data.rename_column('Sent', 'sentence').map(tokenize, batched=True)

    elif dataset_name == 'sst-default':
        data = data.map(tokenize, batched=True).map(lambda batch: {label_name:[int(prob>.5) for prob in batch['label']]}, batched=True)

    elif 'glue-' in dataset_name or dataset_name == 'hans':
        if 'question1' in data.features: # qqp uses this format
            data = data.rename_column('question1', 'sentence1').rename_column('question2', 'sentence2')

        if 'question' in data.features: # qnli uses this format
            data = data.rename_column('question', 'sentence1').rename_column('sentence', 'sentence2')
        
        if 'premise' in data.features: # mnli/hans uses this format
            data = data.rename_column('premise', 'sentence1').rename_column('hypothesis', 'sentence2')
        
        if 'sentence1' in data.features:
            data = data.map(sentence_pair_tokenize, batched=True)
        else:
            data = data.map(tokenize, batched=True)
        data = data.rename_column('label', label_name)
    else:
        data = data.filter(lambda example: len(example['tokens']) > 0)
        data = data.map(seq_tokenize, batched=True)

    number_of_classes = 1
    if subcorpus=='train' and type(data['labels'][0])==list:
        #number_of_classes = len(set([v for seq in data['labels'] for v in seq])) - 1
        number_of_classes = len(data.features['ner_tags'].feature.names)
    elif subcorpus=='train' and dataset_name!='glue-stsb':
        number_of_classes = len(set(data[label_name]))
    to_keep = set(('labels input_ids attention_mask token_type_ids').split())
    data = data.remove_columns([col for col in data.features if col not in to_keep])
    data.set_format("torch")

    if subcorpus=='train':
       data = data.shuffle(seed=42)

    if samples_to_use < 0 or samples_to_use > len(data):
        samples_to_use = len(data)
    elif samples_to_use < 1:
        samples_to_use = int(samples_to_use * len(data))
    data = data.select(range(samples_to_use))

    return data, number_of_classes

def main():
    parser = argparse.ArgumentParser(description='Finetuning experiments.')

    parser.add_argument('--datasets', default='glue-rte glue-mrpc'.split(), nargs='+')
    parser.add_argument('--model', type=str, default='bert-base-cased')
    parser.add_argument('--reference_model', type=str, default=None, required=True)
    parser.add_argument('--tokenizer', type=str, default='bert-base-cased')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use (default:0)')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs to run (default:3)')
    parser.add_argument('--num_experiments', type=int, default=3, help='Epochs to run (default:3)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default:32)')
    parser.add_argument('--grad_accum', type=int, default=1, help='Batch size (default:1)')
    parser.add_argument('--tunable', type=int, default=-1, help='Tunable layers (default:-1)')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate to applay (default:2e-5)')

    special_label_names = {'conll2003': 'ner_tags'}
    args = parser.parse_args()
    if args.reference_model is None:
        args.reference_model = args.tokenizer
    logging.info(args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=512)
    
    num_epochs, gpu_id = args.epochs, args.gpu_id
    device = torch.device("cuda:{}".format(gpu_id)) if torch.cuda.is_available() and gpu_id >= 0 else torch.device("cpu")
    max_layer = 12 if 'base' in args.model else 24
    layers_to_train = ['classifier', 'pooler',] + ['layer.{}'.format(l) for l in range(max_layer - args.tunable, max_layer)]
    logging.info(layers_to_train)

    for i, dataset_name in enumerate(args.datasets):

        label_name = 'labels' if dataset_name not in special_label_names else special_label_names[dataset_name]
        train_data, num_unique_labels = get_dataloaders(dataset_name, tokenizer, 'train', label_name)

        eval_data = []
        if dataset_name.endswith('mnli'):
            for dn, eval_set_name in zip([dataset_name, dataset_name, 'hans'], ['validation_matched', 'validation_mismatched', 'validation']):
                eval_data.append(get_dataloaders(dn, tokenizer, eval_set_name, label_name)[0])
        else:
            eval_set_name = 'test' if dataset_name=='imdb' else 'validation'
            eval_data.append(get_dataloaders(dataset_name, tokenizer, eval_set_name, label_name)[0])
        
        logging.info((dataset_name, len(train_data)))

        for run in range(args.num_experiments):
            random.seed(run)
            np.random.seed(run)
            torch.manual_seed(run)
            torch.cuda.manual_seed(run)
            torch.cuda.manual_seed_all(run)

            if 'conll' in dataset_name.lower():
                reference_model = AutoModelForTokenClassification.from_pretrained(args.reference_model, num_labels=num_unique_labels)
                model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=num_unique_labels)
            elif 'copa' in dataset_name.lower():
                reference_model = AutoModelForMultipleChoice.from_pretrained(args.reference_model)
                model = AutoModelForMultipleChoice.from_pretrained(args.model)
            else:
                reference_model = AutoModelForSequenceClassification.from_pretrained(args.reference_model, num_labels=num_unique_labels)
                model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_unique_labels)
            model.get_parameter('classifier.weight').data.copy_(reference_model.get_parameter('classifier.weight').data.clone())
            model.resize_token_embeddings(len(tokenizer))
            model.to(device)
            logging.info(model.get_parameter('classifier.weight')[0, 0:5])
    
            if args.tunable > -1:
                for p in model.named_parameters():
                    if np.any([x in p[0] for x in layers_to_train]):
                        p[1].requires_grad = True
                    else:
                        p[1].requires_grad = False

            if 'conll' in dataset_name:
                metric = evaluate.load('seqeval')
            else:
                if dataset_name=='NYTK/HuCOLA':
                    metric = evaluate.load('glue', 'cola')
                elif dataset_name.startswith('glue-'):
                    metric = evaluate.load(*dataset_name.split('-'))
                else:
                    metric = evaluate.load('accuracy')

            ed_ids = '-matched -mismatched -hans'.split() if dataset_name.endswith('mnli') else ['' for _ in range(len(eval_data))]
            results, losses = train(model, metric, num_epochs, train_data, eval_data, tokenizer, args.batch_size, args.grad_accum, lr=args.lr, eval_data_ids=ed_ids)

            #logging.info('Finetuning finished')
            #for t in range(len(losses)//50):
            #    logging.info("LOSS\t{}\t{}\t{}\t{:.3f}\t{}".format((t+1)*50, run, dataset_name, np.mean(losses[0:(t+1)*50]), args.model))
            #logging.info("LOSS\t{}\t{}\t{}\t{:.3f}\t{}".format(len(losses), run, dataset_name, np.mean(losses), args.model))

            for ed_num, ed_id in enumerate(ed_ids):
                for epoch, res in enumerate(results[ed_num]):
                    for metric_name, ms in res.items():
                        if type(ms) == dict:
                            for key, val in ms.items():
                                logging.info("RESULTS-{}-{}\t{}\t{}\t{}{}\t{:.4f}\t{}".format(metric_name, key, run, epoch, dataset_name, ed_id, val, args.model))
                        else:
                            logging.info("RESULTS-{}\t{}\t{}\t{}{}\t{:.4f}\t{}".format(metric_name, run, epoch, dataset_name, ed_id, ms, args.model))


if __name__ == '__main__':
    main()
