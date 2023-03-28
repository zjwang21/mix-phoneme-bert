import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
                    prog = 'bpe_pre',
                    description = 'args for preprocessing the data before bpe',)

parser.add_argument('--dic-path', type=str, required=True)
parser.add_argument('--input-path', type=str, required=True)

args = parser.parse_args()

dic = set()
with open(args.input_path, 'r') as f:
    for k in f.readlines():
        k = k.strip().split()
        for c in k: dic.add(c)

with open(args.dic_path, 'w') as f:
    for k in dic:
        f.write(k + ' ' + '1\n')