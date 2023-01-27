import os
import json
import urllib.request
import pandas as pd
from datasets import Dataset
from collections import Counter

URLS = {
        ('HuCOLA', 'train'): 'https://raw.githubusercontent.com/nytud/HuCOLA/32752a757dbecba7c935e6d75641758aeccbfd54/data/cola_train.json',
        ('HuCOLA', 'validation'): 'https://raw.githubusercontent.com/nytud/HuCOLA/32752a757dbecba7c935e6d75641758aeccbfd54/data/cola_dev.json',
        ('HuCOLA', 'test'): 'https://raw.githubusercontent.com/nytud/HuCOLA/32752a757dbecba7c935e6d75641758aeccbfd54/data/cola_test.json',
        ('HuCoPA', 'train'): 'https://raw.githubusercontent.com/nytud/HuCoPA/088bcf06ea16bc62fe4ee0cdbf083d3209236a4c/data/train.json',
        ('HuCoPA', 'val'): 'https://raw.githubusercontent.com/nytud/HuCoPA/088bcf06ea16bc62fe4ee0cdbf083d3209236a4c/data/val.json',
        ('HuCoPA', 'test'): 'https://raw.githubusercontent.com/nytud/HuCoPA/088bcf06ea16bc62fe4ee0cdbf083d3209236a4c/data/test.json',
        ('HuSST', 'train'): 'https://raw.githubusercontent.com/nytud/HuSST/bcf352f37ddd4c5257245adea5defeaf8d1bc148/data/sst_train.json',
        ('HuSST', 'validation'): 'https://raw.githubusercontent.com/nytud/HuSST/bcf352f37ddd4c5257245adea5defeaf8d1bc148/data/sst_dev.json',
        ('HuSST', 'test'): 'https://raw.githubusercontent.com/nytud/HuSST/bcf352f37ddd4c5257245adea5defeaf8d1bc148/data/sst_test.json',
        ('HuWNLI', 'train'): 'https://raw.githubusercontent.com/nytud/HuWNLI/6c79db979d0053511d986c15e0da784f1e33c0eb/data/train.json',
        ('HuWNLI', 'dev'): 'https://raw.githubusercontent.com/nytud/HuWNLI/6c79db979d0053511d986c15e0da784f1e33c0eb/data/dev.json',
        ('HuWNLI', 'test'): 'https://raw.githubusercontent.com/nytud/HuWNLI/6c79db979d0053511d986c15e0da784f1e33c0eb/data/test.json',
        'OpinHuBank': 'https://www.inf.u-szeged.hu/~berendg/docs/OpinHuBank_20130106.csv',
        #'https://hulu.nlp.nytud.hu/{}/{}.json'
        }

def obtain_file(url, out_file_name):
    success = True
    if not os.path.exists(out_file_name):
        try:
            _ = urllib.request.urlretrieve(url, out_file_name)
        except Exception as e:
            print(e, url)
            success = False
    return success

def load_hulu_json(dataset_name, mode):
    out_file_name = '{}_{}.json'.format(dataset_name, mode)
    success = obtain_file(URLS[(dataset_name, mode)], out_file_name)
    data = json.load(open(out_file_name, encoding='utf-8-sig')) if success==True else None
    return data, out_file_name

def aggregate_opinhu_annotations(row, annot_columns):
    c = Counter([row[col] for col in annot_columns])
    class_label, freq = c.most_common(1)[0]
    if freq==2:
        class_label = 0
    return class_label

def load_opinhubank(mode, use_polarized_labels_only):
    out_file_name = 'opinhubank.csv'
    success = obtain_file(URLS['OpinHuBank'], out_file_name)
    data = []
    
    df = pd.read_csv(out_file_name, encoding='iso-8859-2').sample(frac=1.0, random_state=42)
    annot_columns = [col_name for col_name in df.columns if col_name.startswith('Annot')]

    merged_labels = []
    for _, row in df.iterrows():
        merged_labels.append(aggregate_opinhu_annotations(row, annot_columns))

    if use_polarized_labels_only:
        df = df.loc[[l!=0 for l in merged_labels]]

    from_id, to_id = 0, int(0.8*len(df))
    if mode=='validation':
        from_id = to_id
        to_id = int(0.9*len(df))
    elif mode=='test':
        from_id = int(0.9*len(df))
        to_id = len(df)
    elif mode=='all':
        from_id, to_id = 0, len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        if from_id <= i < to_id:
            class_label = aggregate_opinhu_annotations(row, annot_columns)
            if use_polarized_labels_only == False:
                class_label += 1 # {-1,0,+1} => {0, 1, 2}
            elif class_label==-1:
                class_label = 0 # {-1, +1} => {0, +1}

            data.append({'sentence1': row.Sentence, 'sentence2': row.Entity, 'labels': class_label})
    return data, out_file_name

def load_custom_dataset(dataset_name, mode, delete_json=False):
    if 'CoPA' in dataset_name and mode=='validation':
        mode='val'
    elif 'WNLI' in dataset_name and mode=='validation':
        mode='dev'

    if dataset_name.startswith('NYTK/'):
        instances, fn = load_hulu_json(dataset_name.split('/')[1], mode)
    elif 'opinhubank' in dataset_name.lower():
        binary_case = 'bin' in dataset_name.lower()
        instances, fn = load_opinhubank(mode, binary_case)


    if fn is None:
        return None

    if delete_json:
        os.remove(fn)
    
    if 'data' in instances:
        instances = instances['data']

    if 'WNLI' in dataset_name:
        sentence1, sentence2, label = [], [], []
        for instance in instances:
            sentence1.append(instance['sentence1'])
            sentence2.append(instance['sentence2'])
            label.append(int(instance['label']))
        return Dataset.from_dict({'sentence1': sentence1,
                                  'sentence2': sentence2,
                                  'labels': label})
    elif 'CoPA' in dataset_name:
        values = {k:[] for k in instances[0]}
        for instance in instances:
            for k,v in instance.items():
                values[k].append(v if k!='label' else int(v)-1)
        instances = Dataset.from_dict({(k if k!='label' else 'labels'):v for k,v in values.items()})
    elif 'SST' in dataset_name:
        values = {k:[] for k in instances[0]}
        label_mapping = {label:i for i, label in enumerate('negative-neutral-positive'.split('-'))}
        for instance in instances:
            for k,v in instance.items():
                values[k].append(v if k!='Label' else label_mapping[v])
        instances = Dataset.from_dict({(k if k!='Label' else 'labels'):v for k,v in values.items()})
    elif 'HuCOLA' in dataset_name:
        values = {k:[] for k in instances[0]}
        for instance in instances:
            for k,v in instance.items():
                values[k].append(v if k!='Label' else int(v))
        instances = Dataset.from_dict({(k if k!='Label' else 'labels'):v for k,v in values.items()})
    else:
        values = {k:[] for k in instances[0]}
        for instance in instances:
            for k,v in instance.items():
                values[k].append(v)
        instances = Dataset.from_dict(values)

    return instances
