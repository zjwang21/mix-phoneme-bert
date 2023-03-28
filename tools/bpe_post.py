import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
                    prog = 'bpe_post',
                    description = 'args for preprocessing the data after bpe',)

parser.add_argument('--dic-path', type=str, required=True)
parser.add_argument('--input-path', type=str, required=True)
parser.add_argument('--output-path-wwm', type=str, required=True)
parser.add_argument('--output-path-sp', type=str, required=True) 

def read_dic(args):
    with open(args.dic_path, 'r') as f:
        dic = {}
        for k in f.readlines():
            k = k.strip().split()
            dic[k[1]] = k[0]
    return dic

def transform(args, dic):
    with open(args.input_path, 'r') as f, open(args.output_path_wwm, 'w') as fw, open(args.output_path_sp, 'w') as fsp:
        for k in tqdm(f.readlines()):
            wwm = []
            sp = []
            for word in k.strip().split():
                word_post = ''
                word = word.replace('▁', '')
                for c in word: word_post += dic[c]
                wwm.extend(['1'] + ['0'] * (len(word)-1))
                sp.extend([word_post] * len(word))
            fw.write(' '.join(wwm) + '\n')
            fsp.write(' '.join(sp) + '\n')

if __name__ == "__main__":
    args = parser.parse_args()
    dic = read_dic(args)
    transform(args, dic)