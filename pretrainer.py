import os, re, sys, random
import gzip, glob, copy
import spams
import pickle
from tqdm.auto import tqdm
from collections import Counter
from itertools import chain

from sklearn.preprocessing import normalize
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
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelForTokenClassification, ElectraForMaskedLM, ElectraForPreTraining, get_scheduler

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
                 dict_file, kb_file, reinit, hidden_layer, lda, use_amp, rtd, decoupled, dtype, use_masking):

        self.rtd = rtd # replaced token detection
        if dict_file is not None and kb_file is not None:
            logging.warning("Ambiguous parameters (i.e. both a dictionary file and a knowledge base file is given).")
            sys.exit(2)

        self.devices = {}

        conf, conf2 = AutoConfig.from_pretrained(transformer, output_hidden_states=(dict_file is not None and transformer2 is None)), None
        self.rtd = 'electra' in transformer.lower()
        if transformer2 is not None:
            conf2 = AutoConfig.from_pretrained(transformer2, output_hidden_states=dict_file is not None)
            conf.vocab_size = conf2.vocab_size

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.base_model, self.D, self.KB, special_tokens = None, None, None, None
        if reinit == True:
            if self.rtd:
                self.model = ElectraForPreTraining.from_pretrained(transformer)
                self.base_model = ElectraForMaskedLM.from_pretrained(transformer2)
                self.model.resize_token_embeddings(self.tokenizer.vocab_size)
                conf.vocab_size = self.tokenizer.vocab_size
                conf2.vocab_size = self.tokenizer.vocab_size
                conf2.embedding_size = conf.embedding_size

                # medium model config
                #conf.embedding_size = 512
                #if decoupled==False: conf2.embedding_size = 512
                #conf.num_hidden_layers = 8
                #conf.num_attention_heads = 8
                #conf.hidden_size = 512
                #conf.intermediate_size = 2048

                self.model.__init__(conf)
                self.base_model.__init__(conf2)
                if decoupled==False: # in decoupled pre-training, embeddings are not shared, see https://proceedings.mlr.press/v202/dong23c/dong23c.pdf
                    self.base_model.electra.embeddings = self.model.electra.embeddings
                self.set_device(gpu_id2, 'base')
                torch.compile(self.base_model).to(self.devices['base'])
            else:
                self.model = AutoModelForMaskedLM.from_config(conf)
        else:
            if self.rtd:
                self.model = ElectraForPreTraining.from_pretrained(transformer, config=conf)
                self.base_model = ElectraForMaskedLM.from_pretrained(generator)
                self.set_device(gpu_id2, 'base')
                torch.compile(self.base_model).to(self.devices['base'])
            else:
                self.model = AutoModelForMaskedLM.from_pretrained(transformer, config=conf)

        self.layer = hidden_layer
        
        self.masking = use_masking
        self.set_device(gpu_id, 'main')
        torch.compile(self.model).to(self.devices['main'])
        self.amp_settings = {"device_type":self.devices["main"].type,
                             "enabled":use_amp and self.devices['main'].type!='cpu',
                             "dtype": dtype}

        self.train_D = None
        if conf2 is not None and self.rtd==False:
            self.set_device(gpu_id2, 'base')
            self.base_model = AutoModelForMaskedLM.from_pretrained(transformer2, config=conf2)
            torch.compile(self.base_model).to(self.devices['base'])

            if dict_file is not None:
                self.set_device(gpu_id2, 'base')
                self.lasso_params = {'lambda1': lda}
                self.D = torch.from_numpy(np.load(dict_file)).to(self.devices['base'])
                special_tokens = {'additional_special_tokens': [f'[MASK-{i}]' for i in range(self.D.shape[1])]}
        elif dict_file is not None: # this happens when there is no aux model to be used
            self.lasso_params = {'lambda1': lda} 
            self.dl_params = {'K': int(np.abs(int(dict_file))), # abusing the dict_file variable, in this case it refers to the number of dictionary atoms to apply
                              'lambda1': lda, 'posAlpha': True, 'verbose': True, 'iter': 100}
            self.D = 0 # just a temporary placeholder
            self.train_D = False
            special_tokens = {'additional_special_tokens': [f'[MASK-{i}]' for i in range(self.dl_params['K'])]}
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

    def collate(self, sentences, max_seq_length=512, learning_fraction=0.15):
        
        inputs = self.tokenizer(sentences[0], sentences[1], return_tensors="pt", padding=True, return_special_tokens_mask=True,
                                truncation=True, max_length=max_seq_length, return_offsets_mapping=self.KB is not None)
        
        selection_mask = torch.logical_and(torch.rand(inputs['input_ids'].shape) < learning_fraction, inputs['special_tokens_mask']==0)
        special_tokens = inputs['special_tokens_mask']
        del inputs['special_tokens_mask']
        
        inputs['labels'] = -100 * torch.ones_like(inputs['input_ids'])
        inputs['labels'][selection_mask] = inputs['input_ids'][selection_mask]

        num_nnzs = []
        outputs, distributions = None, None
        if self.base_model is not None:
            with torch.autocast(**self.amp_settings):
                if self.rtd: # replaced token detection
                    generator_inputs = copy.deepcopy(inputs)
                    generator_inputs['input_ids'][selection_mask] = self.tokenizer.mask_token_id
                    outputs = self.base_model(**generator_inputs.to(self.devices['base']))
                    distributions = outputs.loss

                    # https://github.com/google-research/electra/blob/8a46635f32083ada044d7e9ad09604742600ee7b/pretrain/pretrain_helpers.py#L222
                    with torch.no_grad():
                        uniform_noise = torch.rand(outputs.logits[selection_mask].shape, device=outputs.logits.device)
                        #logging.info(uniform_noise.shape)
                        gumbel_noise = -torch.log(-torch.log(uniform_noise + 1e-9) + 1e-9)
                        gumbel_noise[:, self.tokenizer.all_special_ids] =- 1000 # we do not want to sample these tokens
                        predictions = torch.argmax(outputs.logits[selection_mask]+gumbel_noise, dim=-1).to(inputs['labels'].device)
                        replacement = inputs['labels'][selection_mask] != predictions
                        inputs['labels'][inputs['attention_mask']==1] = 0 # relevant tokens (assumed to be left intact)
                        inputs['labels'][selection_mask] = replacement.long().to(inputs['labels'].device) # set the labels for the replaced tokens to be 0
                        
                        # apply label smoothing for preventing the discriminator collapsing
                        inputs['labels'] = inputs['labels']#.float()
                        #inputs['labels'][inputs['attention_mask']==1] = 0.9 * inputs['labels'][inputs['attention_mask']==1] + 0.1 * (1 - inputs['labels'][inputs['attention_mask']==1])
                        # alternatively, we might only apply label smoothing towards the replaced label
                        #inputs['labels'][inputs['labels']==1] = 0.9

                        inputs['input_ids'][selection_mask] = predictions.to(inputs['input_ids'].device)

                else:
                    with torch.no_grad():
                        outputs = self.base_model(**inputs.to(self.devices['base']))

        if self.masking and self.rtd==False:
            inputs['input_ids'][selection_mask] = self.tokenizer.mask_token_id
        
        extra_symbols = (self.model.get_input_embeddings().weight.shape[0] - self.tokenizer.vocab_size)

        if self.D is not None and self.base_model is not None:
            with torch.no_grad():
                embeddings = outputs['hidden_states'][self.layer][selection_mask]
                norm = torch.linalg.norm(embeddings, axis=1)
                norm[norm==0] += 1e-9
                embeddings /= norm.reshape(-1,1)
                alphas, _ = sparser.FISTA(embeddings.T, self.D, self.lasso_params['lambda1'], 100)
                
                distributions = torch.nn.functional.normalize(alphas, dim=0, p=1.0).to(self.devices["main"])
        elif outputs is not None and self.rtd==False:
            distributions = outputs['logits'].to(self.devices["main"]) # this is used for distillation
        elif extra_symbols > 0 and self.D is None:
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
    parser = argparse.ArgumentParser(description='Performs pre-training using MLSM (and other objectives).')
    parser.add_argument('--transformer', required=True)
    parser.add_argument('--transformer2')
    parser.add_argument('--tokenizer')

    parser.add_argument('--optimizer', choices='adamw adam lamb'.split(), default='adamw')
    parser.add_argument('--cnt_train', type=str, default=None)

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--gpu_id2', type=int)

    parser.add_argument('--training_seqs', type=int, default=81920000)
    parser.add_argument('--out_dir', default='./checkpoints/')
    parser.add_argument('--dict_file', type=str)
    parser.add_argument('--lda', type=float, default=0.05, help='Controls the sparsity of the LASSO component [0.05].')
    parser.add_argument('--data_location', default='/data2/berend/wiki-bert-pipeline/data/en/filtered-texts/', help='The location of the pretraining data (note that it can be a regexp of multiple files as well).')
    parser.add_argument('--mlm_weight', type=float, default=0.0, help='Controls the weight of the MLM loss [default: 0.0].')
    parser.add_argument('--learning_fraction', type=float, default=0.15, help='The fraction of tokens to learn from (typically by masking them) [default: 0.15].')

    parser.add_argument('--kb_file', type=str)
    parser.add_argument('--max-seq-len', type=int, default=128, help='Maximum sequence length [128].')
    
    parser.add_argument('--reinit', dest='reinit', action='store_true') # whether the model is trained from scratch (if so, the two phase strategy is employed)
    parser.add_argument('--not-reinit', dest='reinit', action='store_false')
    parser.set_defaults(reinit=True)

    parser.add_argument('--kl_loss', dest='kl_loss', action='store_true')
    parser.add_argument('--not-kl_loss', dest='kl_loss', action='store_false')
    parser.set_defaults(kl_loss=True)
    
    parser.add_argument('--special_in', dest='special_input', action='store_true')
    parser.add_argument('--not-special_in', dest='special_input', action='store_false')
    parser.set_defaults(special_input=False)
   
    parser.add_argument('--dtype', type=str, default=None)
    parser.add_argument('--amp', dest='amp', action='store_true')
    parser.add_argument('--not-amp', dest='amp', action='store_false')
    parser.set_defaults(amp=True)

    parser.add_argument('--rtd', dest='rtd', action='store_true') # electra-style replaced token detection task
    parser.add_argument('--not-rtd', dest='rtd', action='store_false')
    parser.set_defaults(rtd=False)
    
    parser.add_argument('--decoupled_rtd', dest='decoupled', action='store_true') # as proposed in https://openreview.net/forum?id=ikE60aXe8M
    parser.add_argument('--not_decoupled_rtd', dest='decoupled', action='store_false')
    parser.set_defaults(decoupled=False)

    parser.add_argument('--use_masking', dest='masking', action='store_true')
    parser.add_argument('--not_use_masking', dest='masking', action='store_false')
    parser.set_defaults(masking=True) # the default is to use masking
    
    parser.add_argument('--layer', type=int, help='Which layers of the model to use during the additional loss computation.')
    parser.add_argument('--batch_size', type=int, help='Number of sentenes per batch. [32]', default=32)
    parser.add_argument('--grad_accum_steps', type=int, help='Number of batches to collect before updates. [1]', default=1)
    parser.add_argument('--lr', type=float, help='The learning rate to apply. [1e-4]', default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, help='Max grad norm used with clipping. [1.0]', default=1.0)
    parser.add_argument('--seed', type=int, help='Random seed to use. [42]', default=42)
    
    args = parser.parse_args()
    if args.gpu_id2 is None:
        args.gpu_id2 = args.gpu_id
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

    if args.dtype == 'bfloat16':
        args.dtype = torch.bfloat16

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    #torch.backends.cuda.matmul.allow_tf32 = True

    p = Pretrainer(args.transformer, args.transformer2, args.tokenizer, args.gpu_id, args.gpu_id2,
                   args.dict_file, args.kb_file, args.reinit, args.layer, args.lda, args.amp, args.rtd, args.decoupled, args.dtype, args.masking)

    hidden_vecs = [] # for storing hidden states of the model. only used in the auxiliary model free variant

    updates_to_perform = args.training_seqs // (args.grad_accum_steps * args.batch_size)
    ce_loss = torch.nn.CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    p.model.train()
    if args.rtd: # in replaced token detection the base model is also simultaneously trained
        p.base_model.train()

    gen_optimizer = None
    if args.optimizer.lower() == 'adamw':
        if p.rtd:
            if args.decoupled:
                gen_optimizer = torch.optim.AdamW(p.base_model.parameters(), lr=args.lr, weight_decay=0.05)
                optimizer = torch.optim.AdamW(p.model.parameters(), lr=3.*args.lr, weight_decay=0.05)
            else:
                optimizer = torch.optim.AdamW(list(p.base_model.parameters()) + [v for k,v in p.model.named_parameters() if 'embedding' not in k], lr=args.lr, weight_decay=0.05)
        else:
            optimizer = torch.optim.AdamW(p.model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'adam':
        if p.rtd:
            if args.decoupled:
                gen_optimizer = torch.optim.Adam(p.base_model.parameters(), lr=args.lr, weight_decay=0.05)
                optimizer = torch.optim.Adam(p.model.parameters(), lr=3.*args.lr, weight_decay=0.05)
            else:
                optimizer = torch.optim.Adam(list(p.base_model.parameters()) + [v for k,v in p.model.named_parameters() if 'embedding' not in k], lr=args.lr, weight_decay=0.05)
        else:
            optimizer = torch.optim.Adam(p.model.parameters(), lr=args.lr)

    elif args.optimizer.lower() == 'lamb': # the optimizer introduced in https://arxiv.org/abs/1904.00962
        if p.rtd:
            if args.decoupled:
                gen_optimizer = torch_optimizer.Lamb(p.base_model.parameters(), lr=args.lr, weight_decay=0.05)
                optimizer = torch_optimizer.Lamb(p.model.parameters(), lr=3.*args.lr, weight_decay=0.05)
            else:
                optimizer = torch_optimizer.Lamb(list(p.base_model.parameters()) + [v for k,v in p.model.named_parameters() if 'embedding' not in k], lr=args.lr, weight_decay=0.05)
        else:
            optimizer = torch_optimizer.Lamb(p.model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    losses, masked_toks, all_nnzs = [], [], []
    lr_scheduler = get_scheduler("linear",
                                 optimizer=optimizer,
                                 num_warmup_steps=updates_to_perform//100,
                                 num_training_steps=updates_to_perform
                                )
    if p.rtd and args.decoupled:
        gen_lr_scheduler = get_scheduler("linear",
                                         optimizer=gen_optimizer,
                                         num_warmup_steps=updates_to_perform//100,
                                         num_training_steps=updates_to_perform)


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

    files = sorted(glob.glob(args.data_location, recursive=True))
    np.random.shuffle(files)

    seqs_covered, updates, updates_to_skip = 0, 0, lr_scheduler.state_dict()['_step_count'] - 1
    checkpoints = set([int(args.training_seqs  // (args.grad_accum_steps * args.batch_size) * percent) for percent in [0.1, 0.25, 0.5, 1.0]])
    logging.info(f'checkpoints: {checkpoints}\ttraining seqs: {args.training_seqs}\taccumulation: {args.grad_accum_steps}\tbatch size: {args.batch_size}\tupdates: {updates}/{updates_to_perform}')
    progress_bar = tqdm(range(updates_to_perform))
   
    loss_cntr = 0
    finished, epoch = False, 0
    batch_text_first, batch_text_second = [], [] # the first sentences and their continuations
    first_sentence = True
    while not finished:
      epoch += 1
      logging.info(f"EPOCH {epoch}")
      for input_file in files:
        logging.info(f"Input file: {input_file}")
        if finished: break
        fo = gzip.open(input_file, 'rt', encoding='utf-8') if input_file.endswith('.gz') else open(input_file, encoding='utf-8')

        for line in fo:
            line = line.rstrip()
            if len(line) < 5 and first_sentence: # do not process such a sentence that we would drop later anyway
                continue

            if first_sentence:
                batch_text_first.append(line)
                first_sentence = False
            else:
                batch_text_second.append(line)
                first_sentence = True

                if len(batch_text_second[-1]) < 5 or len(batch_text_first[-1].split()) + len(batch_text_second[-1].split()) < 5:
                    popped_sentenceA = batch_text_first.pop()
                    popped_sentenceB = batch_text_second.pop()
                    #logging.info(f"popped sentences: {popped_sentenceA} {popped_sentenceB}")


            if len(batch_text_second)==args.batch_size:

                loss_cntr += 1
                seqs_covered += len(batch_text_second)
                if updates >= updates_to_skip:
                    batch, nnzs, mask, expected_distro = p.collate([batch_text_first, batch_text_second], max_seq_length=args.max_seq_len, learning_fraction=args.learning_fraction)
                    #for k,v in batch.items():
                    #    print(k, v, v.shape)
                    #    print("=======")
                    #    if k=='input_ids': print (p.tokenizer.decode(v[0]))
                    #print(mask)
                    #print(expected_distro)
                    #print(p.base_model.training)
                    #sys.exit(2)
                    all_nnzs.extend(nnzs)
                    masked_toks.append(mask.sum().item())
                
                    #b = {k: batch[k].to(p.devices['main']) for k in ['token_type_ids', 'attention_mask', 'labels', 'input_ids'] if k in batch} 
                    if args.special_input:
                        batch['input_ids'][mask] = torch.argmax(expected_distro, dim=0) + p.tokenizer.vocab_size
                        expected_distro = None
                    with torch.autocast(**p.amp_settings):
                        outputs = p.model(**batch)
                    
                    if p.train_D is not None and expected_distro is None and 'hidden_states' in outputs: # this happens in the auxiliary model free case
                        hidden_vecs.append((outputs['hidden_states'][args.layer][batch['attention_mask']==1]).detach().cpu().numpy().T)
                        hidden_vecs = hidden_vecs[-200:]
                        if p.train_D:
                            with torch.no_grad():
                                embeddings = outputs['hidden_states'][args.layer][mask]
                                norm = torch.linalg.norm(embeddings, axis=1)
                                norm[norm==0] += 1e-9
                                embeddings /= norm.reshape(-1,1)
                                alphas, _ = sparser.FISTA(embeddings.T, p.D, p.lasso_params['lambda1'], 100)
                                expected_distro = torch.nn.functional.normalize(alphas, dim=0, p=1.0).to(p.devices["main"])
                        outputs['hidden_states'] = None # for saving some extra memory

                    kl_loss_val, ce_loss_val = None, torch.tensor(0.0, device=p.model.device)
                    if p.rtd==False and expected_distro is not None:
                        if p.spec_symbol_mapping is not None:
                            if args.kl_loss:
                                predicted_log_distr = torch.nn.functional.log_softmax(outputs.logits[mask][:, p.tokenizer.vocab_size:], dim=1)
                                kl_loss_val = kl_loss(predicted_log_distr, expected_distro.T)
                            else:
                                max_positions = torch.argmax(expected_distro, dim=0)
                                ce_loss_val = ce_loss(outputs.logits[mask].view(-1, len(p.tokenizer))[:, p.tokenizer.vocab_size:], max_positions)
                        else: # standard distillation without special symbols
                            predicted_log_distr = torch.nn.functional.log_softmax(outputs.logits[mask], dim=1)
                            target_distro = torch.nn.functional.softmax(expected_distro[mask], dim=1)
                            kl_loss_val = kl_loss(predicted_log_distr, target_distro)
                        if args.mlm_weight > 0:
                            ce_loss_val += args.mlm_weight * ce_loss(outputs.logits.view(-1, len(p.tokenizer))[:, 0:p.tokenizer.vocab_size], batch['labels'].view(-1))
                    else:
                        if p.rtd:
                            ce_loss_multiplier = 1. if args.decoupled else 50.
                            ce_loss_val = ce_loss_multiplier * outputs.loss # this is the binary loss
                            kl_loss_val = expected_distro # this is an ugly hack, i.e., the MLM loss of the generator is handled at this point
                            
                            nan_loss = torch.isnan(ce_loss_val) or torch.isnan(kl_loss_val)
                            am = batch['attention_mask']
                            label_sum = batch['labels'][am==1].sum()
                            #targets = batch['labels'][am==1]
                            #probs = 1 / (1+torch.exp(-outputs.logits[am==1]))
                            #own_losses = -torch.log(probs) * targets - torch.log(1-probs) * (1-targets)
                            #own_losses *= ce_loss_multiplier
                            #logging.info(own_losses)
                            if ce_loss_val.item() > ce_loss_multiplier * 0.65 or updates%100==0 or nan_loss:
                                logging.info(f'{updates}\t{ce_loss_multiplier}\t{ce_loss_val.item():.5f}\t{kl_loss_val.item():.5f}\t{am.sum()}\t{label_sum}')
                                alert = False # torch.max(own_losses) > ce_loss_multiplier * 4.
                                #logging.info(f'OWN loss calc:\t{ce_loss_val.item()}\t{torch.mean(own_losses)}\t{torch.max(own_losses)}\t{alert}')
                                confusion = np.zeros((2,2))
                                M = batch['labels'] != -100
                                for seq_id in range(outputs.logits.shape[0]):

                                    if alert or (ce_loss_val > ce_loss_multiplier * 0.6 and updates > 100):
                                        logging.info(f'============== SEQ ID {seq_id} @ update {updates} ===============')
                                        logging.info(f'SUSPICIOUS\t{updates}\t{seq_id}\t{batch_text_first[seq_id]}\t{batch_text_second[seq_id]}')
                                    #inputs = batch['input_ids'][seq_id]
                                    #logging.info(p.tokenizer.decode(inputs, skip_special_tokens=True))
                                    m = M[seq_id]
                                    for pred_logit, truth, token_id in zip(outputs.logits[seq_id][m], batch['labels'][seq_id][m], batch['input_ids'][seq_id][m]):
                                        if alert or (ce_loss_val > ce_loss_multiplier * 0.6 and updates > 100):
                                            prob = 1 / (1+torch.exp(-pred_logit))
                                            warning = (prob > 0.99 and truth.item()==False) or (prob<0.01 and truth.item()==True)
                                            logging.info(f'SUSPICIOUS\t{updates}\t{pred_logit.item() > 0}\t{truth.item()}\t{p.tokenizer.decode([token_id])}\t{pred_logit.item():.4f}\t{prob:.3f}\t{"PROBLEM" if warning else "OK"}')
                                        confusion[1 if pred_logit > 0 else 0, 1 if truth > 0.5 else 0] += 1
                                    #print(batch['labels'][seq_id][M])
                                    #print(batch['input_ids'][seq_id][M])
                                if updates%100==0:
                                    logging.info(f'CONFUSION\t{updates}\t{confusion}\t{np.sum(np.diag(confusion))/np.sum(confusion)}\t{ce_loss_val.item()}\t{kl_loss_val.item()}')
                                    #logging.info(torch.linalg.norm(p.model.get_parameter('electra.embeddings.word_embeddings.weight')))
                                    #logging.info(torch.linalg.norm(p.base_model.get_parameter('electra.embeddings.word_embeddings.weight')))
                            if nan_loss:
                                p.model.save_pretrained(f'{args.out_dir}/{updates}')
                                p.tokenizer.save_pretrained(f'{args.out_dir}/{updates}')
                                if p.rtd:
                                    p.base_model.save_pretrained(f'{args.out_dir}/{updates}/generator')
                                    p.tokenizer.save_pretrained(f'{args.out_dir}/{updates}/generator')
                                sys.exit(2)
                        else:
                            ce_loss_val = ce_loss(outputs.logits.view(-1, len(p.tokenizer))[:, 0:p.tokenizer.vocab_size], batch['labels'].view(-1))
                    losses.append((ce_loss_val.item(), 0. if kl_loss_val is None else kl_loss_val.item()))

                    if kl_loss_val is not None and ce_loss_val > 0: # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers
                        if p.rtd:
                            scaler.scale(kl_loss_val / args.grad_accum_steps).backward() #retain_graph=True)
                            scaler.scale(ce_loss_val / args.grad_accum_steps).backward()

                            #scaler.scale((kl_loss_val + ce_loss_val) / args.grad_accum_steps).backward()
                            #for k,v in p.model.named_parameters():
                            #    print(k, torch.linalg.norm(v.grad))
                            #print("============")
                            #for k,v in p.base_model.named_parameters():
                            #    print(k, torch.linalg.norm(v.grad))
                            #sys.exit(2)
                        else:
                            scaler.scale((kl_loss_val + ce_loss_val) / args.grad_accum_steps).backward()
                    elif kl_loss_val is not None:
                        scaler.scale(kl_loss_val / args.grad_accum_steps).backward() # normalization for gradient accumulation
                    elif ce_loss_val > 0:
                        scaler.scale(ce_loss_val / args.grad_accum_steps).backward() # normalization for gradient accumulation

                if loss_cntr % args.grad_accum_steps == 0:
                    if updates==0:
                        logging.info((len(batch_text_first), len(batch_text_second), batch_text_first[0], batch_text_second[0], batch_text_first[-1]))
                    
                    if updates >= updates_to_skip:
                        if args.max_grad_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(p.model.parameters(), args.max_grad_norm)
                            if p.rtd:
                                if gen_optimizer is not None:
                                    scaler.unscale_(gen_optimizer)
                                torch.nn.utils.clip_grad_norm_(p.base_model.parameters(), args.max_grad_norm)

                        scaler.step(optimizer)
                        if gen_optimizer is not None:
                            scaler.step(gen_optimizer)
                        scaler.update()
                        #optimizer.step()
                        optimizer.zero_grad()
                        if gen_optimizer is not None:
                            gen_optimizer.zero_grad()
                            gen_lr_scheduler.step()
                            nan_error = False
                            for k,v in p.model.named_parameters():
                                if torch.any(torch.isnan(v)):
                                    logging.error(f'update{updates} DISC: NaN for {k}')
                                    nan_error = True
                            for k,v in p.base_model.named_parameters():
                                if torch.any(torch.isnan(v)):
                                    logging.error(f'update{updates} GEN: NaN for {k}')
                                    nan_error = True
                            if nan_error:
                                p.model.save_pretrained(f'{args.out_dir}/{updates}')
                                p.tokenizer.save_pretrained(f'{args.out_dir}/{updates}')
                                p.base_model.save_pretrained(f'{args.out_dir}/{updates}/generator')
                                p.tokenizer.save_pretrained(f'{args.out_dir}/{updates}/generator')
                        lr_scheduler.step()

                    progress_bar.update(1)
                    updates += 1
                    
                    if p.train_D is not None and updates >= 10000 and updates % 10000 == 0:
                        X = normalize(np.hstack(hidden_vecs), axis=0)
                        logging.info((X.shape, np.linalg.norm(X, axis=0).shape, np.linalg.norm(X, axis=0)))
                        #self.dl_params['verbose'] = True
                        old_D = torch.clone(p.D) if p.train_D else 0
                        p.D = spams.trainDL(X, D=np.asfortranarray(p.D.cpu().numpy()) if p.train_D else None, **p.dl_params)
                        p.D = torch.from_numpy(p.D).to(p.devices['main'])
                        logging.info(f"Dict. diff Fro. norm after {updates} updates: {torch.linalg.norm(old_D - p.D):.4f}")
                        hidden_vecs = []
                        p.train_D = True
                
                    if updates >=updates_to_skip and (updates==1 or updates % (min(checkpoints) // 100) == 0):
                        logging.info("LOSSES\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.8f}\t{}\t{:.1f}\t{}".format(input_file, seqs_covered, updates, np.mean([l[0] for l in losses]), np.mean([l[1] for l in losses]), *lr_scheduler.get_last_lr(), masked_toks[-1], np.mean(masked_toks), scaler.get_scale()))
                        if len(all_nnzs) > 0:
                            logging.info(f"NNZ\t{seqs_covered}\t{updates}\t{np.mean(all_nnzs):.4f}\t{len(all_nnzs)}")
                    if updates > 0 and updates in checkpoints:
                        logging.info((len(batch_text_first), len(batch_text_second), batch_text_first[0], batch_text_second[0], batch_text_first[-1]))
                        logging.info(f'Checkpoint created after {updates} updates')
                        p.model.save_pretrained(f'{args.out_dir}/{updates}')
                        p.tokenizer.save_pretrained(f'{args.out_dir}/{updates}')
                        if p.train_D:
                            np.save(f'{args.out_dir}/{updates}/dict', p.D.cpu().numpy())
                        if p.rtd:
                            p.base_model.save_pretrained(f'{args.out_dir}/{updates}/generator')
                            p.tokenizer.save_pretrained(f'{args.out_dir}/{updates}/generator')

                        #pickle.dump({'scheduler': lr_scheduler.state_dict(), 'optim': optimizer.state_dict(), 'losses': losses, 'masked_toks': masked_toks, 'all_nnzs': all_nnzs}, open(f'{args.out_dir}/{updates}/optim_state_dict.pkl', 'wb'))

                    if updates == updates_to_perform:
                        finished = True
                        break

                batch_text_first, batch_text_second = [], []
                first_sentence = True
