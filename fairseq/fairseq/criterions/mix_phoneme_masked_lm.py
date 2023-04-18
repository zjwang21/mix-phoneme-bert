# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import math
from omegaconf import II

import torch
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class MixPhonemeMaskedLmConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")


@register_criterion("mix_phoneme_masked_lm", dataclass=MixPhonemeMaskedLmConfig)
class MixPhonemeMaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg: MixPhonemeMaskedLmConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens_sp = sample["target"]["sp_tgt_tokens"].ne(self.padding_idx)
        masked_tokens_p = sample["target"]["p_tgt_tokens"].ne(self.padding_idx)
        sample_size = masked_tokens_sp.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_tokens_sp = None  # always project all tokens on TPU
        elif masked_tokens_sp.device == torch.device("cpu"):
            if not masked_tokens_sp.any():
                masked_tokens_sp = None
                masked_tokens_p = None
        else:
            masked_tokens_sp = torch.where(
                masked_tokens_sp.any(),
                masked_tokens_sp,
                masked_tokens_sp.new([True]),
            )
            masked_tokens_p = torch.where(
                masked_tokens_p.any(),
                masked_tokens_p,
                masked_tokens_p.new([True]),
            )
        targets = sample["target"]
        if masked_tokens_sp is not None:
            targets_sp = targets['sp_tgt_tokens'][masked_tokens_sp]
        if masked_tokens_p is not None:
            targets_p = targets['p_tgt_tokens'][masked_tokens_p]

        logits = model(**sample["net_input"], masked_tokens=masked_tokens_sp, targets_sp=targets_sp)[0]
        logits_p = logits['phoneme'].view(-1, logits['phoneme'].size(-1))
        logits_sp = logits['sup-phoneme'].view(-1, logits['sup-phoneme'].size(-1))

        targets_p = targets_p.view(-1)
        targets_sp = logits['targets_sp'].view(-1)

        sample_size_sp = targets_sp.size(0)
        phonemes = torch.argmax(torch.softmax(logits_p, dim=-1), dim=-1)
        sup_phonemes = torch.argmax(torch.softmax(logits_sp, dim=-1), dim=-1)
        acc_p = (phonemes == targets_p).sum()
        acc_sp = (sup_phonemes == targets_sp).sum()

        loss_p = modules.cross_entropy(
            logits_p,
            targets_p,
            reduction="sum",
            ignore_index=self.padding_idx,
        )
        loss_sp = modules.cross_entropy(
            logits_sp,
            targets_sp,
            reduction="sum",
            ignore_index=self.padding_idx,
        )
        loss = (loss_p + loss_sp) / 2.0
        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "loss_p": loss_p if self.tpu else loss_p.data,
            "loss_sp": loss_sp if self.tpu else loss_sp.data,
            "acc_p": acc_p,
            "acc_sp": acc_sp,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "sample_size_sp": sample_size_sp,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_sum_p = sum(log.get("loss_p", 0) for log in logging_outputs)
        loss_sum_sp = sum(log.get("loss_sp", 0) for log in logging_outputs)
        acc_sum_p = sum(log.get("acc_p", 0) for log in logging_outputs)
        acc_sum_sp = sum(log.get("acc_sp", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        sample_size_sp = sum(log.get("sample_size_sp", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_p", loss_sum_p / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_sp", loss_sum_sp / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "acc_phoneme", acc_sum_p / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "acc_sup_phoneme", acc_sum_sp / sample_size_sp / math.log(2), sample_size_sp, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
