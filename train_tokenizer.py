import os, argparse
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import AlbertTokenizerFast

def main():
    parser = argparse.ArgumentParser(description='Training a tokenizer')
    parser.add_argument('--vocab_size', type=int, default=25000, help='Vocab size (default:25000)')
    parser.add_argument('--folder', default='/srv/data/berend/babylm/babylm_data/babylm_10M/')
    parser.add_argument('--out_folder')
    parser.add_argument('--cased', dest='cased', action='store_true')
    parser.add_argument('--uncased', dest='cased', action='store_false')
    parser.set_defaults(cased=True)

    args = parser.parse_args()
    tokenizer = Tokenizer(models.Unigram())

    normalizers_list = [normalizers.Replace("``", '"'), normalizers.Replace("''", '"')]
    if args.cased == False:
        normalizers_list.append(normalizers.Lowercase())
    tokenizer.normalizer = normalizers.Sequence(normalizers_list)
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

    trainer = trainers.UnigramTrainer(vocab_size=args.vocab_size, special_tokens=["[CLS]", "[SEP]", "<unk>", "<pad>", "[MASK]"], unk_token="<unk>")
    tokenizer.train([f'{args.folder}/{fn}' for fn in os.listdir(args.folder)], trainer=trainer)

    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS]:0 $A:0 [SEP]:0",
        pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )
    tokenizer.decoder = decoders.Metaspace()

    new_tokenizer = AlbertTokenizerFast(tokenizer_object=tokenizer)
    out_path = args.out_folder
    if out_path is None:
        out_path = f'{args.folder}/{args.vocab_size}_cased{args.cased}'
    new_tokenizer.save_pretrained(out_path)

if __name__ == '__main__':
    main()
