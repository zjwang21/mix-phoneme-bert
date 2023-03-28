tools=/home/wangzhijun2/mix-phoneme-bert/tools

python $tools/preprocess.py --input-path ./data.txt --output-dir ./bpe --test-size 1000
