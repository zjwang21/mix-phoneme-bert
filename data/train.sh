DATA_DIR=/home/wangzhijun2/mix-phoneme-bert/data/bpe/bin

fairseq-hydra-train -m --config-dir /home/wangzhijun2/mix-phoneme-bert/fairseq/examples/roberta/config/pretraining \
--config-name base task.data=$DATA_DIR
