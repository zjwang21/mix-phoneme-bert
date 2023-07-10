root=/home/u1190303311/mix-phoneme-bert-main/data/bpe
tools=/home/u1190303311/mix-phoneme-bert-main/tools
spm_train=/home/u1190303311/mix-phoneme-bert-main/fairseq/scripts/spm_train.py
spm_encode=/home/u1190303311/mix-phoneme-bert-main/fairseq/scripts/spm_encode.py

python $tools/bpe_pre.py --dic-path $root/dict_for_bpe.txt \
    --input-path $root/train.p \
    --output-path $root/train_for_bpe.txt \

python $spm_train --input=$root/train_for_bpe.txt \
    --model_prefix=$root/spm \
    --vocab_size=1000 \
    --model_type=bpe \
    --input_sentence_size=10000 \
    --shuffle_input_sentence=true

mkdir $root/zh/

for split in train test valid
do

python $tools/bpe_pre.py --dic-path $root/dict_for_bpe.txt \
    --input-path $root/$split.p \
    --output-path $root/zh/$split.p \

python $spm_encode --model=$root/spm.model < $root/zh/$split.p > $root/zh/$split.bpe.p

python $tools/bpe_post.py --dic-path $root/dict_for_bpe.txt \
    --input-path $root/zh/$split.bpe.p \
    --output-path-wwm $root/$split.wwm \
    --output-path-sp $root/$split.sp
done

python $tools/generate_dict.py --input-path $root/train.sp --dic-path $root/dict_sp.txt
