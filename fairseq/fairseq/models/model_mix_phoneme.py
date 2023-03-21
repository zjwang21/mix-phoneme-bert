# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder, MixPhonemeTransformerEncoder
from fairseq.modules import LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import safe_getattr, safe_hasattr

from .roberta.model import RobertaModel, RobertaLMHead

logger = logging.getLogger(__name__)

@register_model("mix_phoneme_roberta")
class MixPhonemeRobertaModel(RobertaModel):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        from omegaconf import OmegaConf

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, False)

        # make sure all arguments are present
        mix_phoneme_base_architecture(args)

        if not safe_hasattr(args, "max_positions"):
            if not safe_hasattr(args, "tokens_per_sample"):
                args.tokens_per_sample = task.max_positions()
            args.max_positions = args.tokens_per_sample

        encoder = MixPhonemeRobertaEncoder(args, task.source_dictionary)

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        return cls(args, encoder)


class MixPhonemeRobertaEncoder(FairseqEncoder):
    """MixPhonemeRoBERTa encoder."""

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        dictionary_p, dictionary_sp = dictionary
        # set any missing default values
        mix_phoneme_base_architecture(args)
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        embed_tokens_p = self.build_embedding(
            len(dictionary_p), args.encoder_embed_dim, dictionary_p.pad()
        )

        embed_tokens_sp = self.build_embedding(
            len(dictionary_sp), args.encoder_embed_dim, dictionary_sp.pad()
        )

        self.sentence_encoder = self.build_encoder(args, dictionary, (embed_tokens_p, embed_tokens_sp))

        self.lm_head_p = self.build_lm_head(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary_p),
            activation_fn=args.activation_fn,
            weight=(
                self.sentence_encoder.embed_tokens_p.weight
                if not args.untie_weights_roberta
                else None
            ),
        )
        self.lm_head_sp = MixPhonemeRobertaLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary_sp),
            activation_fn=args.activation_fn,
            weight=(
                self.sentence_encoder.embed_tokens_sp.weight
                if not args.untie_weights_roberta
                else None
            ),
        )

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = MixPhonemeTransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        targets_sp=None,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens
        )
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens, targets_sp=targets_sp)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        return features, {"inner_states": inner_states}

    def output_layer(self, features, masked_tokens=None, targets_sp=None, **unused):
        logits_sp, target_sp = self.lm_head_sp(features, masked_tokens=masked_tokens, targets_sp=targets_sp)
        return {'phoneme': self.lm_head_p(features, masked_tokens), 
                'sup-phoneme': logits_sp,
                'targets_sp': target_sp}

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

class MixPhonemeRobertaLMHead(nn.Module):
    """Head for mix-phoneme masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, targets_sp=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        logits_sp = features.view(-1, features.size(-1))
        targets_sp = targets_sp.view(-1)
        temp = torch.cat((targets_sp.new([-1]), targets_sp[:-1]))
        word_begins = temp == targets_sp
        word_begins_ids = torch.argwhere(word_begins == False).T[0]
        word_begins_ids = torch.cat((word_begins_ids, word_begins_ids.new([targets_sp.size(-1)])))
        word_lens = word_begins_ids[1:] - word_begins_ids[:-1]
        logits_sp_tuple = torch.split(logits_sp, word_lens.tolist(), dim=0)
        targets_sp = targets_sp.masked_select(~word_begins)
        assert targets_sp.size(-1) == len(logits_sp_tuple)
        logits_sp = torch.stack([torch.mean(x, dim=0) for x in logits_sp_tuple], dim=0)

        x = self.dense(logits_sp)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x, targets_sp


@register_model_architecture("mix_phoneme_roberta", "mix_phoneme_roberta")
def mix_phoneme_base_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 12)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.0)

    args.max_source_positions = safe_getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = safe_getattr(args, "encoder_learned_pos", True)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)
    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", True)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = safe_getattr(
        args, "encoder_normalize_before", False
    )
    args.pooler_activation_fn = safe_getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_roberta = safe_getattr(args, "untie_weights_roberta", False)

    # Adaptive input config
    args.adaptive_input = safe_getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = safe_getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = safe_getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = safe_getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = safe_getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = safe_getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = safe_getattr(
        args, "spectral_norm_classification_head", False
    )


@register_model_architecture("mix_phoneme_roberta", "mix_phoneme_roberta_prenorm")
def mix_phoneme_roberta_prenorm_architecture(args):
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", False)
    args.encoder_normalize_before = safe_getattr(args, "encoder_normalize_before", True)
    mix_phoneme_base_architecture(args)


@register_model_architecture("mix_phoneme_roberta", "mix_phoneme_roberta_base")
def mix_phoneme_roberta_base_architecture(args):
    mix_phoneme_base_architecture(args)


@register_model_architecture("mix_phoneme_roberta", "mix_phoneme_roberta_large")
def mix_phoneme_roberta_large_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
    mix_phoneme_base_architecture(args)


@register_model_architecture("mix_phoneme_roberta", "mix_phoneme_xlm")
def mix_phoneme_xlm_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 16)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1280)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 1280 * 4)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
    mix_phoneme_base_architecture(args)
