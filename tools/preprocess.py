import numpy as np
import argparse
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(
                    prog = 'bpe_pre',
                    description = 'args for preprocessing the data before bpe',)

parser.add_argument('--input-path', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)
parser.add_argument('--test-size', type=int, required=True)

def read_input(path):
    with open(path, 'r') as f:
        return f.readlines()

def write(data, ids, path):
    with open(path, 'w') as f:
        for k in ids:
            f.write(data[k])

def train_test_split(p, test_size, args):
    #ids = torch.randperm(len(sp))
    ids = list(range(len(p)))
    valid_ids = ids[:test_size]
    test_ids = ids[test_size:test_size * 2]
    train_ids = ids[test_size * 2:]
    write(p, train_ids, os.path.join(args.output_dir, 'train.p'))
    write(p, test_ids, os.path.join(args.output_dir, 'test.p'))
    write(p, valid_ids, os.path.join(args.output_dir, 'valid.p'))


def get_dict(path):
    with open(path, 'r') as f:
        data = set()
        for k in f.readlines():
            for s in k.strip().split():
                data.add(s)
    return list(data)


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    p_res = read_input(args.input_path)
    print('spliting with test size {}'.format(args.test_size))
    train_test_split(p_res, args.test_size, args)
    p_dict = get_dict(os.path.join(args.output_dir, 'train.p'))
    with open(os.path.join(args.output_dir, 'dict_p.txt'), 'w') as f:
        for k in p_dict:
            f.write(k + ' ' + '1' + '\n')

    with open(os.path.join(args.output_dir, 'dict_mask.txt'), 'w') as f:
        f.write('0 1\n')
        f.write('1 1\n')
    
    with open(os.path.join(args.output_dir, 'dict_for_bpe.txt'), 'w') as f:
        begin = 0x4e00
        for k in p_dict:
            f.write(k + ' ' + chr(begin) + '\n')
            begin+=1