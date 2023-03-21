from fairseq.models.model_mix_phoneme import MixPhonemeRobertaModel
import torch
#assert isinstance(roberta.model, torch.nn.Module)
from fairseq.data import indexed_dataset, Dictionary
from fairseq import tasks
from fairseq.tasks.mix_phoneme_masked_lm import MixPhonemeMaskedLMConfig, MixPhonemeMaskedLMTask
from tqdm import tqdm
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print('use gpu: ', torch.cuda.is_available())
print('device count: ', torch.cuda.device_count())
#print('device name index 0: ', torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = MixPhonemeMaskedLMConfig()
config.data = '/mnt/lustre/wangzhijun2/mix-bert/data/data_TTS/bin'
config._name = 'mix_phoneme_masked_lm'
task = MixPhonemeMaskedLMTask.setup_task(config)

dataset_p, dataset_sp = task.build_dataset_for_embedding('test')

output_dir = '/mnt/lustre/share_data/wangzhijun2/mix-bert-embeddings/'
ckpt_dir = '/mnt/lustre/wangzhijun2/mix-bert/data/5kw/multirun/2023-01-15/20-12-15/0/checkpoints/'
data_path = '/mnt/lustre/wangzhijun2/mix-bert/data/data_TTS/bin'
roberta = MixPhonemeRobertaModel.from_pretrained(ckpt_dir, 'checkpoint_best.pt', data_path, bpe=None)

print('[Loading model from {}]'.format(ckpt_dir + 'checkpoint_best.pt'))
print('[Loading data from {}]'.format(data_path))

ids = []
with open('/mnt/lustre/wangzhijun2/mix-bert/data/data_TTS/src.txt', 'r') as f:
    for k in f.readlines():
        k = k.strip().split('|||')
        ids.append(k[0].strip())

roberta.to(device)
for id, sp, p, in tqdm(zip(ids, dataset_sp, dataset_p)):
    src_tokens = {'sp_src_tokens': sp.unsqueeze(0).to(device), 
                  'p_src_tokens': p.unsqueeze(0).to(device)} ##device

    with torch.no_grad():
        output = roberta.model(src_tokens, features_only=True)[0]
    embedding = output[0].cpu().detach().numpy()        #squeeze
    embedding = embedding[1:-1]

    assert len(sp) == len(p)
    assert embedding.shape == (len(sp)-2, 256), '{}'.format(embedding.shape)
    np.save(output_dir + id, embedding)
