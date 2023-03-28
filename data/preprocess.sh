path=/home/wangzhijun2/mix-phoneme-bert/data/bpe
for lang in p sp
do
fairseq-preprocess \
    --only-source \
    --source-lang $lang --target-lang $lang \
    --srcdict $path/dict_${lang}.txt \
    --trainpref $path/train \
    --validpref $path/valid \
    --testpref $path/test \
    --destdir $path/bin \
    --workers 20
done

for lang in wwm
do
fairseq-preprocess \
    --only-source \
    --source-lang $lang --target-lang $lang \
    --srcdict $path/dict_mask.txt \
    --trainpref $path/train \
    --validpref $path/valid \
    --testpref $path/test \
    --destdir $path/bin \
    --workers 20
done
