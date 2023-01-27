import os, re
import sys, random
import gzip
from tqdm.auto import tqdm
from collections import Counter

from nltk.corpus import wordnet as wn
try:
    wn.get_version()
except:
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')

import numpy as np
import sparser

import torch
import torch_optimizer
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForMaskedLM, get_scheduler

import argparse
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

class Pretrainer(object):

    def __init__(self, transformer, transformer2, tokenizer, gpu_id, gpu_id2,
                 dict_file, kb_file, reinit, hidden_layer, lda):
        if dict_file is not None and kb_file is not None:
            logging.warning("Ambiguous parameters (i.e. both a dictionary file and a knowledge base file is given).")
            sys.exit(2)

        self.devices = {}

        conf, conf2 = AutoConfig.from_pretrained(transformer, output_hidden_states=False), None
        if transformer2 is not None:
            conf2 = AutoConfig.from_pretrained(transformer2, output_hidden_states=(dict_file is not None))
            conf.vocab_size = conf2.vocab_size
        
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        if reinit == True:
            self.model = AutoModelForMaskedLM.from_config(conf)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(transformer, config=conf)
        #self.model.half()

        self.layer = hidden_layer
        
        self.set_device(gpu_id, 'main')
        self.model.to(self.devices['main'])

        self.base_model, self.D, self.KB, special_tokens = None, None, None, None
        if conf2 is not None:
            self.set_device(gpu_id2, 'base')
            self.base_tokenizer = AutoTokenizer.from_pretrained(transformer2)
            self.base_model = AutoModelForMaskedLM.from_pretrained(transformer2, config=conf2)
            self.base_model.to(self.devices['base'])

            if dict_file is not None:
                self.set_device(gpu_id2, 'base')
                self.lasso_params = {'lambda1': lda}
                self.D = torch.from_numpy(np.load(dict_file)).to(self.devices['base'])
                special_tokens = {'additional_special_tokens': ['[MASK-{}]'.format(i) for i in range(self.D.shape[1])]}
        elif kb_file is not None:
            if os.path.exists(kb_file):
                relations = []
                for i,l in enumerate(gzip.open(kb_file, 'rt')):
                    rel, x, y = l.split()[1:4]
                    if x[0:6]==y[0:6]=='/c/en/' and not '_' in x:
                        relations.append(('{}_{}'.format(rel, re.sub('/.*', '', y[6:])), re.sub('/.*', '', x[6:])))
                common_rels = Counter([r[0] for r in relations]).most_common(3000)
                special_tokens = {'additional_special_tokens': ['[{}]'.format(r[0]) for r in common_rels]}

                common_rels_set = set([cr[0] for cr in common_rels])
                self.KB = {}
                for i, (rel, word) in enumerate(relations):
                    if rel in common_rels_set:
                        if word not in self.KB: self.KB[word] = []
                        self.KB[word].append(rel)
            else:
                lexnames = list(sorted(set([s.lexname() for s in wn.all_eng_synsets()])))
                special_tokens = {'additional_special_tokens': ['[{}]'.format(ln) for ln in lexnames]}
        
        if special_tokens:
            logging.info("Tokens added: {}".format(len(special_tokens['additional_special_tokens'])))
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.spec_symbol_mapping = dict(zip(self.tokenizer.additional_special_tokens, self.tokenizer.additional_special_tokens_ids))
        else:
            logging.info("Num tokens: {}".format(len(self.tokenizer)))
            self.spec_symbol_mapping = None
            self.model.resize_token_embeddings(len(self.tokenizer))

    def set_device(self, device_id, name):
        device_count = torch.cuda.device_count()
        if device_count != 0 and device_id >= 0:
            if device_id >= device_count:
                device_id = random.randint(1, device_count) - 1
            self.devices[name] = torch.device('cuda:{}'.format(device_id))
        else:
            self.devices[name] = torch.device('cpu')


    def collate(self, sentences, max_seq_length=512):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_seq_length, return_offsets_mapping=self.KB is not None).to(self.devices['main'])
        outputs = None
        if self.base_model is not None:
            with torch.no_grad():
                base_inputs = self.base_tokenizer(sentences, return_tensors="pt", padding=True,
                                                  truncation=True, max_length=max_seq_length, return_offsets_mapping=self.KB is not None).to(self.devices['base'])
                outputs = self.base_model(**base_inputs)

        selection_mask = torch.zeros_like(inputs['input_ids'], dtype=bool)
        inputs['labels'] = -100 * torch.ones_like(inputs['input_ids'], device=self.devices['main'])
        for i, att_mask in enumerate(inputs['attention_mask']):
            num_tokens = att_mask.sum().item()
            selected_positions = sorted(1+np.random.choice(range(num_tokens-2), int(num_tokens*.15), replace=False))
            selection_mask[i, selected_positions] = True

        inputs['labels'][selection_mask] = inputs['input_ids'][selection_mask]
        inputs['input_ids'][selection_mask] = self.tokenizer.mask_token_id
        
        # out of the initially masked subwords choose a fraction to be masked specially
        # mask2 = selection_mask * (torch.rand_like(selection_mask, dtype=float) < np.abs(self.replace_ratio))
        #inputs['special_vals'] = inputs['labels'][mask2].clone()

        num_nnzs, distributions = [], None
        extra_symbols = (self.model.get_input_embeddings().weight.shape[0] - self.tokenizer.vocab_size)

        if self.D is not None:
            with torch.no_grad():
                embeddings = outputs['hidden_states'][self.layer][selection_mask]
                norm = torch.linalg.norm(embeddings, axis=1)
                norm[norm==0] += 1e-9
                embeddings /= norm.reshape(-1,1)
                alphas, _ = sparser.FISTA(embeddings.T, self.D, self.lasso_params['lambda1'], 100)
                
                distributions = torch.nn.functional.normalize(alphas, dim=0, p=1.0).to(self.devices['main'])
        elif outputs is not None:
            distributions = outputs['logits'] # this is used for distillation
        elif extra_symbols > 0:
            distributions = torch.zeros((len(self.spec_symbol_mapping), selection_mask.sum()), device=self.devices['main'])
            for i, token_id in enumerate(inputs['labels'][selection_mask]):
                token = self.tokenizer.decode(token_id).lower()
                features = []
                if self.KB is not None and token in self.KB:
                    features = self.KB[token]
                elif self.KB is None:
                    features = [s.lexname() for s in wn.synsets(token)]
                for feature in features:
                    property_id = self.spec_symbol_mapping['[{}]'.format(feature)] - self.tokenizer.vocab_size
                    distributions[property_id, i] += 1.0
            distributions = torch.nn.functional.normalize(distributions, dim=0, p=1.0)
        
        return inputs, num_nnzs, selection_mask, distributions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs pre-training using BERTsparse.')
    parser.add_argument('--transformer', required=True)
    parser.add_argument('--transformer2')
    parser.add_argument('--tokenizer')

    parser.add_argument('--optimizer', choices='adamw lamb'.split(), required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--gpu_id2', type=int, default=0)

    parser.add_argument('--training_seqs', type=int, default=81920000)
    parser.add_argument('--out_dir', default='./checkpoints/')
    parser.add_argument('--dict_file', type=str)
    parser.add_argument('--lda', type=float, default=0.05, help='Controls the sparsity of the LASSO component.')
    parser.add_argument('--data_location', default='/data2/berend/wiki-bert-pipeline/data/en/filtered-texts/')

    parser.add_argument('--kb_file', type=str)
    parser.add_argument('--max-seq-len', type=int, default=512)
    
    parser.add_argument('--reinit', dest='reinit', action='store_true') # whether the model is trained from scratch (if so, the two phase strategy is employed)
    parser.add_argument('--not-reinit', dest='reinit', action='store_false')
    parser.set_defaults(reinit=True)

    parser.add_argument('--kl_loss', dest='kl_loss', action='store_true')
    parser.add_argument('--not-kl_loss', dest='kl_loss', action='store_false')
    parser.set_defaults(kl_loss=True)
    
    parser.add_argument('--special_in', dest='special_input', action='store_true')
    parser.add_argument('--not-special_in', dest='special_input', action='store_false')
    parser.set_defaults(special_input=False)
    
    parser.add_argument('--layer', type=int, help='Which layers of the model to use during the additional loss computation.')
    parser.add_argument('--batch_size', type=int, help='Number of sentenes per batch. [32]', default=32)
    parser.add_argument('--grad_accum_steps', type=int, help='Number of batches to collect before updates. [1]', default=1)
    parser.add_argument('--lr', type=float, help='The learning rate to apply. [2e-5]', default=2e-5)
    parser.add_argument('--max_grad_norm', type=float, help='Max grad norm used with clipping. [1.0]', default=1.0)
    
    args = parser.parse_args()
    logging.info(args)

    if args.dict_file is None and args.kb_file is None:
        args.kl_loss = False
    
    if os.path.exists(args.out_dir):
        logging.warning('Out directory already exists')
        sys.exit(2)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    #torch.backends.cuda.matmul.allow_tf32 = True

    p = Pretrainer(args.transformer, args.transformer2, args.tokenizer, args.gpu_id, args.gpu_id2,
                   args.dict_file, args.kb_file, args.reinit, args.layer, args.lda)

    updates_to_perform = args.training_seqs // (args.grad_accum_steps * args.batch_size)
    ce_loss = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    p.model.train()

    if args.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(p.model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'lamb': # the optimizer introduced in https://arxiv.org/abs/1904.00962
        optimizer = torch_optimizer.Lamb(p.model.parameters(), lr=args.lr) 

    lr_scheduler = get_scheduler("linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=updates_to_perform//100,
                                 num_training_steps=updates_to_perform
                                )
    progress_bar = tqdm(range(updates_to_perform))

    if os.path.isdir(args.data_location):
        files = sorted(['{}/{}/{}'.format(args.data_location, d, f) for d in os.listdir(args.data_location) for f in os.listdir('{}/{}/'.format(args.data_location, d))])
    else:
        files = [args.data_location]
    np.random.shuffle(files)

    losses = []
    updates, seqs_covered = 0, 0
    checkpoints = set([int(args.training_seqs  // (args.grad_accum_steps * args.batch_size) * percent) for percent in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]])
    logging.info('checkpoints: {}\ttraining seqs: {}\taccumulation: {}\tbatch size: {}\tupdates: {}'.format(checkpoints, args.training_seqs, args.grad_accum_steps, args.batch_size, updates_to_perform))
   
    first_phase = args.reinit # the part of training with shorter max seq length and larger batches
    factor_in_first_phase = 4
    finished, epoch = False, 0
    batch_text, all_nnzs, masked_toks = [], [], []
    while not finished:
      epoch += 1
      logging.info("EPOCH {}".format(epoch))
      for input_file in files:
        if finished: break
        fo = gzip.open(input_file, 'rt', encoding='utf-8') if input_file.endswith('.gz') else open(input_file, encoding='utf-8')
        #logging.info(input_file)

        for line in fo:
            line = line.rstrip()
            if len(line) < 5 or len(line.split()) < 5: continue

            batch_text.append(line)
            if len(batch_text)==((factor_in_first_phase if first_phase else 1) * args.batch_size):

                max_seq_len = args.max_seq_len
                if first_phase:
                    max_seq_len //= factor_in_first_phase

                batch, nnzs, mask, expected_distro = p.collate(batch_text, max_seq_length=max_seq_len)
                all_nnzs.extend(nnzs)
                masked_toks.append(mask.sum().item())
                
                b = {k: batch[k].to(p.devices['main']) for k in ['token_type_ids', 'attention_mask', 'labels', 'input_ids']} 
                if args.special_input:
                    b['input_ids'][mask] = torch.argmax(expected_distro, dim=0) + p.tokenizer.vocab_size
                    expected_distro = None
                outputs = p.model(**b)

                if expected_distro is not None:
                    if p.spec_symbol_mapping is not None:
                        if args.kl_loss:
                            predicted_log_distr = torch.nn.functional.log_softmax(outputs.logits[mask][:, p.tokenizer.vocab_size:], dim=1)
                            loss_val = kl_loss(predicted_log_distr, expected_distro.T)
                        else:
                            max_positions = torch.argmax(expected_distro, dim=0)
                            loss_val = ce_loss(outputs.logits[mask].view(-1, len(p.tokenizer))[:, p.tokenizer.vocab_size:], max_positions)
                    else: # standard distillation without special symbols
                        predicted_log_distr = torch.nn.functional.log_softmax(outputs.logits[mask], dim=1)
                        target_distro = torch.nn.functional.softmax(expected_distro[mask], dim=1)
                        loss_val = kl_loss(predicted_log_distr, target_distro)
                else:
                    loss_val = ce_loss(outputs.logits.view(-1, len(p.tokenizer))[:, 0:p.tokenizer.vocab_size], b['labels'].view(-1))
                losses.append(loss_val.item())

                accum_size = args.grad_accum_steps // (factor_in_first_phase if first_phase else 1)

                loss = loss_val / accum_size # normalization for gradient accumulation
                loss.backward()
                seqs_covered += len(batch_text)

                if len(losses) % accum_size == 0:
                    if updates > 200:
                        last_losses = np.mean(losses[-accum_size])
                        if last_losses > 1.1 * np.mean(losses):
                            logging.info("WARNING after {} updates: {:.4f} vs {:.4f} {:.7f}".format(updates, last_losses, np.mean(losses), *lr_scheduler.get_last_lr()))
                    if args.max_grad_norm > 0:
                        # grad clipping taken from the transformers.Trainer implementation
                        if hasattr(optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(p.model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            p.model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling full precision
                            torch.nn.utils.clip_grad_norm_(p.model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    progress_bar.update(1)
                    updates += 1
                    first_phase = args.reinit and updates < 0.9 * updates_to_perform
                
                    if updates==1 or updates % (min(checkpoints) // 10) == 0:
                        logging.info("LOSSES\t{}\t{}\t{}\t{:.4f}\t{:.8f}\t{}\t{:.1f}\t{}".format(input_file, seqs_covered, updates, np.mean(losses), *lr_scheduler.get_last_lr(), masked_toks[-1], np.mean(masked_toks), first_phase))
                        if len(all_nnzs) > 0:
                            logging.info("NNZ\t{}\t{}\t{:.4f}\t{}".format(seqs_covered, updates, np.mean(all_nnzs), len(all_nnzs)))
                    if updates > 0 and updates in checkpoints:
                        p.model.save_pretrained('{}/{}'.format(args.out_dir, updates))
                    if updates == updates_to_perform:
                        finished = True
                        break

                batch_text = []
