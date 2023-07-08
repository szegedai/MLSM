import argparse, os, sys, json, gzip, random
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
    parser.add_argument('--transformer', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--corpus_location', help='A (gzipped) file or a dictionary of files')
    parser.add_argument('--num_dict_atoms', type=int, default='3000', help='The number of semantic atoms in the learned dictionary matrix (default: 3000)')
    parser.add_argument('--output_path', default='./out')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use (default:0)')
    parser.add_argument('--tokens_used', type=int, default=1000000, help='Subtokens to collect (default:1000000)')


    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.gpu_id)) if torch.cuda.is_available() and args.gpu_id >= 0 else torch.device("cpu")

    conf = AutoConfig.from_pretrained(args.transformer, output_hidden_states=True)
    m = AutoModel.from_pretrained(args.transformer, config=conf).to(device)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    total_layers = m.config.num_hidden_layers
    embeddings = [] #{total_layers-final_layer:[] for final_layer in range(4)}
    sentences_selected = 0
    selected_embeddings = []
    with gzip.open(args.corpus_location, 'rt') as fi:
        for li,l in enumerate(fi):
            if len(l.split()) > 4 and len(selected_embeddings) < args.tokens_used:
                token_tensor = {k:v.to(m.device) for k,v in tok(l.strip(), return_tensors='pt', max_length=512, truncation=True).items()}
                relevant_tokens = token_tensor['input_ids'].shape[1] - 1 # we leave the [CLS] out
                #selected_token_positions = np.random.choice(range(1,relevant_tokens), size=int(0.2*relevant_tokens), replace=False)
                selected_token_positions = list(range(1,relevant_tokens))
                if len(selected_token_positions) > 0:
                    sentences_selected += 1
                    selected_embeddings.extend([tok.decode(token_tensor['input_ids'][0][t]) for t in selected_token_positions])
                    if sentences_selected%2000==0:
                        logging.info((sentences_selected, len(selected_embeddings), selected_token_positions, l, selected_embeddings[-20:]))
                    with torch.no_grad():
                        outputs = m(**token_tensor)
                        embeddings.extend(outputs['hidden_states'][args.layer][0, selected_token_positions].cpu().numpy())

    c = Counter(selected_embeddings)
    to_keep = [c[se] / len(selected_embeddings) < 1e-3 for se in selected_embeddings]
    json.dump(list(zip(selected_embeddings,to_keep)), open(f'{args.output_path}/selected_embeddings.json', 'w'))
    logging.info((len(selected_embeddings), sum(to_keep), sentences_selected))
    #for l, vecs in embeddings.items():
    #    np.save(f'{args.output_path}/embeddings_{l}', np.array(vecs))
    logging.info(selected_embeddings[0:80])

    #e = json.load(open(f'{args.output_path}/selected_embeddings.json'))

    #for l, vecs in embeddings.items():
    X = np.array([embeddings[r] for r in np.random.permutation(range(len(embeddings)))])
    E = normalize(X).T
    '''
        embs, tokens, v = [], [], None
        X = np.load(f'{args.output_path}/embeddings_{l}.npy')
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
        
        threshold = np.inf
        E = normalize(np.array([em for t, em in zip(tokens, embs) if c[t] < threshold]), copy=False).T
    '''

    params = {'K': args.num_dict_atoms, 'lambda1': 0.05, 'numThreads': 8, 'iter': 1000, 'batchsize': 400, 'posAlpha': True, 'verbose': False}

    logging.info(E.shape)
    D = spams.trainDL(E, **params)
    np.save(f'{args.output_path}/layer_{args.layer}_k_{args.num_dict_atoms}', D)
