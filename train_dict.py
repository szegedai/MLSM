import argparse, os, sys, json, gzip, glob, random
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
    parser.add_argument('--tokenizer')
    parser.add_argument('--max_tokens', type=int, default=128)
    parser.add_argument('--relative_layer', type=int, default=0, help="Which layer to use relative to the last layer (default:0).")
    parser.add_argument('--corpus_location', help='A (gzipped) file or a dictionary of files')
    parser.add_argument('--num_dict_atoms', type=int, default='3000', help='The number of semantic atoms in the learned dictionary matrix (default: 3000)')
    parser.add_argument('--output_path', help="Location to save the dictionary matrix")
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use (default:0)')
    parser.add_argument('--tokens_used', type=int, default=1000000, help='Subtokens to collect (default:1000000)')

    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.gpu_id)) if torch.cuda.is_available() and args.gpu_id >= 0 else torch.device("cpu")

    conf = AutoConfig.from_pretrained(args.transformer, output_hidden_states=True)
    m = AutoModel.from_pretrained(args.transformer, config=conf).to(device)
    if args.tokenizer is None:
        tok = AutoTokenizer.from_pretrained(args.transformer)
    else:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)

    if args.output_path is None:
        args.output_path = args.transformer

    args.layer = m.config.num_hidden_layers + args.relative_layer
    logging.info(args)
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    files = sorted(glob.glob(args.corpus_location, recursive=True))
    np.random.shuffle(files)

    total_layers = m.config.num_hidden_layers
    embeddings = [] #{total_layers-final_layer:[] for final_layer in range(4)}
    sentences_selected = 0
    sentence_pair = []
    selected_embeddings = []
    for input_file in files:
        fi = gzip.open(input_file, 'rt', encoding='utf-8') if input_file.endswith('.gz') else open(input_file, encoding='utf-8')
        for li,l in enumerate(fi):
            if len(l.split()) > 4:
                sentence_pair.append(l.strip())
            else:
                sentence_pair = []

            if len(sentence_pair) == 2 and len(selected_embeddings) < args.tokens_used:
                token_tensor = {k:v.to(m.device) for k,v in tok(sentence_pair[0], sentence_pair[1], return_tensors='pt', max_length=args.max_tokens, truncation=True).items()}
                sentence_pair = []
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

    t = 1e-5 # threshold to be applied for pruning (if any)
    c = Counter(selected_embeddings)
    to_keep_dict = {'none': [True for se in selected_embeddings]}
    to_keep_dict['w2v'] = [bool(1 - np.sqrt(t / (c[se] / len(selected_embeddings))) < np.random.rand()) for se in selected_embeddings]
    to_keep_dict['drop'] = [c[se] / len(selected_embeddings) < t for se in selected_embeddings]

    #for l, vecs in embeddings.items():
    #    np.save(f'{args.output_path}/embeddings_{l}', np.array(vecs))
    logging.info(selected_embeddings[0:80])
    
    for pruning_strategy, to_keep in to_keep_dict.items():
        logging.info((pruning_strategy, type(to_keep), len(selected_embeddings), sum(to_keep), sentences_selected))
        json.dump(list(zip(selected_embeddings, to_keep)), open(f'{args.output_path}/selected_embeddings_{pruning_strategy}.json', 'w'))

        embeddings_reduced = [e for e,decision in zip(embeddings, to_keep) if decision]
        X = np.array([embeddings[r] for r in np.random.permutation(range(len(embeddings)))])
        E = normalize(X).T

        params = {'K': args.num_dict_atoms, 'lambda1': 0.05, 'numThreads': 8, 'iter': 1000, 'batchsize': 400, 'posAlpha': True, 'verbose': False}

        logging.info((pruning_strategy, E.shape))
        D = spams.trainDL(E, **params)
        np.save(f'{args.output_path}/{args.transformer.replace("/", "_")}_layer_{args.layer}_k_{args.num_dict_atoms}_{pruning_strategy}', D)
