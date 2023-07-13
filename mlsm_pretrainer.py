import os, re, shutil
import sys, random
import gzip
import pickle
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
                 dict_file, kb_file, reinit, hidden_layer, lda, use_amp):
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
            conf.vocab_size = self.tokenizer.vocab_size
            self.model = AutoModelForMaskedLM.from_config(conf)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(transformer, config=conf)

        self.layer = hidden_layer
        
        self.set_device(gpu_id, 'main')
        torch.compile(self.model).to(self.devices['main'])
        self.amp_settings = {"device_type":self.devices["main"].type, "enabled":use_amp and self.devices['main'].type!='cpu'}

        self.base_model, self.D, self.KB, special_tokens = None, None, None, None
        if conf2 is not None:
            self.set_device(gpu_id2, 'base')
            #self.base_tokenizer = AutoTokenizer.from_pretrained(transformer2)
            self.base_model = AutoModelForMaskedLM.from_pretrained(transformer2, config=conf2)
            torch.compile(self.base_model).to(self.devices['base'])

            if dict_file is not None:
                self.set_device(gpu_id2, 'base')
                self.lasso_params = {'lambda1': lda}
                self.D = torch.from_numpy(np.load(dict_file)).to(self.devices['base'])
                special_tokens = {'additional_special_tokens': [f'[MASK-{i}]' for i in range(self.D.shape[1])]}
        elif kb_file is not None:
            if os.path.exists(kb_file):
                relations = []
                for i,l in enumerate(gzip.open(kb_file, 'rt')):
                    rel, x, y = l.split()[1:4]
                    if x[0:6]==y[0:6]=='/c/en/' and not '_' in x:
                        relations.append((f"{rel}_{ re.sub('/.*', '', y[6:])}", re.sub('/.*', '', x[6:])))
                common_rels = Counter([r[0] for r in relations]).most_common(3000)
                special_tokens = {'additional_special_tokens': [f'[{r[0]}]'  for r in common_rels]}

                common_rels_set = set([cr[0] for cr in common_rels])
                self.KB = {}
                for i, (rel, word) in enumerate(relations):
                    if rel in common_rels_set:
                        if word not in self.KB: self.KB[word] = []
                        self.KB[word].append(rel)
            else:
                lexnames = list(sorted(set([s.lexname() for s in wn.all_eng_synsets()])))
                special_tokens = {'additional_special_tokens': [f'[{ln}]' for ln in lexnames]}
        
        if special_tokens:
            logging.info(f"Tokens added: {len(special_tokens['additional_special_tokens'])}")
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.spec_symbol_mapping = dict(zip(self.tokenizer.additional_special_tokens, self.tokenizer.additional_special_tokens_ids))
        else:
            logging.info(f"Num tokens: {len(self.tokenizer)}")
            self.spec_symbol_mapping = None
            self.model.resize_token_embeddings(len(self.tokenizer))

    def set_device(self, device_id, name):
        device_count = torch.cuda.device_count()
        if device_count != 0 and device_id >= 0:
            if device_id >= device_count:
                device_id = random.randint(1, device_count) - 1
            self.devices[name] = torch.device(f'cuda:{device_id}')
        else:
            self.devices[name] = torch.device('cpu')

    def collate(self, sentences, max_seq_length=512):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_seq_length, return_offsets_mapping=self.KB is not None)
        outputs = None
        if self.base_model is not None:
            with torch.no_grad():
                #base_inputs = self.base_tokenizer(sentences, return_tensors="pt", padding=True,
                base_inputs = self.tokenizer(sentences, return_tensors="pt", padding=True,
                                                  truncation=True, max_length=max_seq_length, return_offsets_mapping=self.KB is not None).to(self.devices['base'])
                with torch.autocast(**self.amp_settings):
                    outputs = self.base_model(**base_inputs)

        selection_mask = torch.zeros_like(inputs['input_ids'], dtype=bool)
        inputs['labels'] = -100 * torch.ones_like(inputs['input_ids'])
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
                
                distributions = torch.nn.functional.normalize(alphas, dim=0, p=1.0).to(self.devices["main"])
        elif outputs is not None:
            distributions = outputs['logits'].to(self.devices["main"]) # this is used for distillation
        elif extra_symbols > 0:
            distributions = torch.zeros((len(self.spec_symbol_mapping), selection_mask.sum()))
            for i, token_id in enumerate(inputs['labels'][selection_mask]):
                token = self.tokenizer.decode(token_id).lower()
                features = []
                if self.KB is not None and token in self.KB:
                    features = self.KB[token]
                elif self.KB is None:
                    features = [s.lexname() for s in wn.synsets(token)]
                for feature in features:
                    property_id = self.spec_symbol_mapping[f'[{feature}]'] - self.tokenizer.vocab_size
                    distributions[property_id, i] += 1.0
            distributions = torch.nn.functional.normalize(distributions, dim=0, p=1.0).to(self.devices["main"])
        
        return inputs.to(self.devices['main']), num_nnzs, selection_mask, distributions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs pre-training using BERTsparse.')
    parser.add_argument('--transformer', required=True)
    parser.add_argument('--transformer2')
    parser.add_argument('--tokenizer')

    parser.add_argument('--optimizer', choices='adamw lamb'.split(), required=True)
    parser.add_argument('--cnt_train', type=str, default=None)

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--gpu_id2', type=int, default=0)

    parser.add_argument('--training_seqs', type=int, default=81920000)
    parser.add_argument('--first_phase_seqs', type=int, default=-1, help='The number of training steps with a limited sequence length. If negative all steps are of limited lengt.')
    parser.add_argument('--out_dir', default='./checkpoints/')
    parser.add_argument('--dict_file', type=str)
    parser.add_argument('--lda', type=float, default=0.05, help='Controls the sparsity of the LASSO component.')
    parser.add_argument('--data_location', default='/data2/berend/wiki-bert-pipeline/data/en/filtered-texts/')
    parser.add_argument('--mlm_weight', type=float, default=0.0, help='Controls the weight of the MLM loss [default: 0.0].')

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
    
    parser.add_argument('--amp', dest='amp', action='store_true')
    parser.add_argument('--not-amp', dest='amp', action='store_false')
    parser.set_defaults(amp=True)
    
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
        if args.cnt_train is None:
            sys.exit(2)

    if args.cnt_train is not None and args.reinit:
        logging.error('The continued training and weight reinitalization settings contradict each other.')
        sys.exit(2)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    #torch.backends.cuda.matmul.allow_tf32 = True

    p = Pretrainer(args.transformer, args.transformer2, args.tokenizer, args.gpu_id, args.gpu_id2,
                   args.dict_file, args.kb_file, args.reinit, args.layer, args.lda, args.amp)

    updates_to_perform = args.training_seqs // (args.grad_accum_steps * args.batch_size)
    ce_loss = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    p.model.train()

    if args.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(p.model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'lamb': # the optimizer introduced in https://arxiv.org/abs/1904.00962
        optimizer = torch_optimizer.Lamb(p.model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    losses, masked_toks, all_nnzs = [], [], []
    lr_scheduler = get_scheduler("linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=updates_to_perform//100,
                                 num_training_steps=updates_to_perform
                                )
    if args.cnt_train is not None:
        if os.path.exists(args.cnt_train):
            optim_states = pickle.load(open(args.cnt_train, 'rb'))
            optimizer.load_state_dict(optim_states['optim'])
            lr_scheduler.load_state_dict(optim_states['scheduler'])
            losses = optim_states['losses']
            masked_toks = optim_states['masked_toks']
            all_nnzs = optim_states['all_nnzs']
            logging.info(lr_scheduler)
        else:
            logging.error('Optimizer state path specified, but does not exist.')
            sys.exit(2)

    if os.path.isdir(args.data_location):
        files = sorted([f'{args.data_location}/{d}/{f}' for d in os.listdir(args.data_location) for f in os.listdir(f'{args.data_location}/{d}/')])
    else:
        files = [args.data_location]
    np.random.shuffle(files)

    seqs_covered, updates, updates_to_skip = 0, 0, lr_scheduler.state_dict()['_step_count'] - 1
    checkpoints = set([int(args.training_seqs  // (args.grad_accum_steps * args.batch_size) * percent) for percent in [0.2, 0.4, 0.6, 0.8, 1.0]])
    logging.info(f'checkpoints: {checkpoints}\ttraining seqs: {args.training_seqs}\taccumulation: {args.grad_accum_steps}\tbatch size: {args.batch_size}\tupdates: {updates}/{updates_to_perform}')
    progress_bar = tqdm(range(updates_to_perform))
   
    first_phase = True # boolean indicating if it is the part of training with shorter max seq length and larger batches
    factor_in_first_phase = 4
    loss_cntr = 0
    finished, epoch = False, 0
    batch_text = []
    while not finished:
      epoch += 1
      logging.info(f"EPOCH {epoch}")
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

                accum_size = args.grad_accum_steps // (factor_in_first_phase if first_phase else 1)
                loss_cntr += 1
                seqs_covered += len(batch_text)
                if updates >= updates_to_skip:
                    batch, nnzs, mask, expected_distro = p.collate(batch_text, max_seq_length=max_seq_len)
                    all_nnzs.extend(nnzs)
                    masked_toks.append(mask.sum().item())
                
                    #b = {k: batch[k].to(p.devices['main']) for k in ['token_type_ids', 'attention_mask', 'labels', 'input_ids'] if k in batch} 
                    if args.special_input:
                        batch['input_ids'][mask] = torch.argmax(expected_distro, dim=0) + p.tokenizer.vocab_size
                        expected_distro = None
                    with torch.autocast(**p.amp_settings):
                        outputs = p.model(**batch)

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
                        if args.mlm_weight > 0:
                            loss_val += args.mlm_weight * ce_loss(outputs.logits.view(-1, len(p.tokenizer))[:, 0:p.tokenizer.vocab_size], batch['labels'].view(-1))
                    else:
                        loss_val = ce_loss(outputs.logits.view(-1, len(p.tokenizer))[:, 0:p.tokenizer.vocab_size], batch['labels'].view(-1))
                    losses.append(loss_val.item())

                    scaler.scale(loss_val / accum_size).backward() # normalization for gradient accumulation

                if loss_cntr % accum_size == 0:
                    if updates==0:
                        logging.info((len(batch_text), batch_text[0], batch_text[-1]))
                    
                    if updates >= updates_to_skip:
                        if args.max_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(p.model.parameters(), args.max_grad_norm)

                        scaler.step(optimizer)
                        scaler.update()
                        #optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()

                    progress_bar.update(1)
                    updates += 1
                    first_phase = args.first_phase_seqs<0 or updates < args.first_phase_seqs
                
                    if updates >=updates_to_skip and (updates==1 or updates % (min(checkpoints) // 100) == 0):
                        logging.info("LOSSES\t{}\t{}\t{}\t{:.4f}\t{:.8f}\t{}\t{:.1f}\t{}\t{}".format(input_file, seqs_covered, updates, np.mean(losses), *lr_scheduler.get_last_lr(), masked_toks[-1], np.mean(masked_toks), scaler.get_scale(), first_phase))
                        if len(all_nnzs) > 0:
                            logging.info(f"NNZ\t{seqs_covered}\t{updates}\t{np.mean(all_nnzs):.4f}\t{len(all_nnzs)}")
                    if updates > 0 and updates in checkpoints:
                        logging.info((len(batch_text), batch_text[0], batch_text[-1]))
                        logging.info(f'Checkpoint created after {updates} updates')
                        p.model.save_pretrained(f'{args.out_dir}/{updates}')
                        #pickle.dump({'scheduler': lr_scheduler.state_dict(), 'optim': optimizer.state_dict(), 'losses': losses, 'masked_toks': masked_toks, 'all_nnzs': all_nnzs}, open(f'{args.out_dir}/{updates}/optim_state_dict.pkl', 'wb'))
                        if os.path.isdir(args.tokenizer):
                            for tf in os.listdir(args.tokenizer):
                                shutil.copy(f'{args.tokenizer}/{tf}', f'{args.out_dir}/{updates}/{tf}')
                    if updates == updates_to_perform:
                        finished = True
                        break

                batch_text = []
