DATA_DIR=/home/u1190303311/mix-phoneme-bert-main/data/bpe/bin

fairseq-hydra-train -m --config-dir /home/u1190303311/mix-phoneme-bert-main/fairseq/examples/roberta/config/pretraining \
--config-name base task.data=$DATA_DIR
