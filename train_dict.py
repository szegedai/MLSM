import os, sys, json, gzip
import argparse
import numpy as np
from collections import Counter
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer, AutoConfig
import spams, torch
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performs dictionary learning of hidden representations.')
    parser.add_argument('--transformer', default='SZTAKI-HLT/hubert-base-cc')
    parser.add_argument('--corpus_location', help='A gzipped file')
    parser.add_argument('--num_dict_atoms', default='3000', help='The number of semantic atoms in the learned dictionary matrix')

    conf = AutoConfig.from_pretrained(args.transformer, output_hidden_states=True)
    m = AutoModel.from_pretrained(args.transformer, config=conf).to('cuda')
    tok = AutoTokenizer.from_pretrained(args.transformer)

    np.random.seed(42)

    embeddings = {layer:[] for layer in range(9,13)}
    sentences_selected = 0
    selected_embeddings = []
    with gzip.open(args.corpus_location, 'rt') as fi:
        for li,l in enumerate(fi):
            if len(l.split()) > 4 and sentences_selected < 100000:
                token_tensor = {k:v.to(m.device) for k,v in tok(l, return_tensors='pt', max_length=512, truncation=True).items()}
                relevant_tokens = token_tensor['input_ids'].shape[1] - 1 # we leave the [CLS] out
                #selected_token_positions = np.random.choice(range(1,relevant_tokens), size=int(0.2*relevant_tokens), replace=False)
                selected_token_positions = list(range(1,relevant_tokens))
                if len(selected_token_positions) > 0:
                    sentences_selected += 1
                    selected_embeddings.extend([tok.decode(token_tensor['input_ids'][0][t]) for t in selected_token_positions])
                    if sentences_selected%2000==0:
                        logging.info((sentences_selected, len(selected_embeddings), selected_token_positions))
                    with torch.no_grad():
                        outputs = m(**token_tensor)
                        for l in embeddings:
                            embeddings[l].extend(outputs['hidden_states'][l][0, selected_token_positions].cpu().numpy())

    c = Counter(selected_embeddings)
    to_keep = [c[se] / len(selected_embeddings) < 1e-3 for se in selected_embeddings]
    json.dump(list(zip(selected_embeddings,to_keep)), open('selected_embeddings.json', 'w'))
    logging.info((len(selected_embeddings), sum(to_keep), sentences_selected))
    for k,vecs in embeddings.items():
        np.save('embeddings_{}'.format(k), np.array(vecs))


    e = json.load(open('selected_embeddings.json'))

    for l in range(9,13):
        X = np.load('embeddings_{}.npy'.format(l))

        embs, tokens, v = [], [], None
        for (token, common), emb in zip(e, X):
            if token.startswith('##'):
                n+=1
                v+=emb
                tokens[-1] += token[2:]
            else:
                if v is not None:
                    embs.append(v / n)
                tokens.append(token)
                n = 1
                v = emb
        embs.append(v / n)
        c = Counter(tokens)
        threshold = sum(c.values())*.001
        E = normalize(np.array([em for t, em in zip(tokens, embs) if c[t] < threshold]), copy=False).T

        params = {'K': args.num_dict_atoms, 'lambda1': 0.05, 'numThreads': 8, 'iter': 1000, 'batchsize': 400, 'posAlpha': True, 'verbose': False}

        logging.info(E.shape)
        D = spams.trainDL(E, **params)
        np.save('layer_{}_k_{}'.format(l, args.num_ditc_atoms), D)
