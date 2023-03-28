import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
                    prog = 'bpe_pre',
                    description = 'args for preprocessing the data before bpe',)

parser.add_argument('--dic-path', type=str, required=True)
parser.add_argument('--input-path', type=str, required=True)
parser.add_argument('--output-path', type=str, required=True)

def transform(args):
    with open(args.input_path, 'r') as f:
        data = []
        for k in f.readlines(): data.append(k.strip())
    with open(args.dic_path, 'r') as f:
        dic = {}
        for k in f.readlines():
            k = k.strip().split()
            dic[k[0]] = k[1]

    with open(args.output_path, 'w') as f:
        for k in tqdm(data):
            cur = ''
            for c in k.strip().split(): cur += dic[c]
            f.write(cur + '\n')


if __name__ == "__main__":
    args = parser.parse_args()
    transform(args)