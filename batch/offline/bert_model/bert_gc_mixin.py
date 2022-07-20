
import poptorch
import logging
logger = logging.getLogger()

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from typing import List, Optional, Tuple
import torch
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads: Set[int] = set()

    def prune_heads(self, heads: List[int]):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, attention_head_size, self.pruned_heads)
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)
        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = 1000*(mask - 1).to(torch.half)
        mask = mask.view(mask_reshp).expand_as(scores)
        #print("A", mask.shape, scores.shape)
        scores = scores + mask

        #mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        #scores = scores.masked_fill(mask, torch.tensor(-1000.0,dtype=torch.half))  # (bs, n_heads, q_length, k_length)

        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)




def logger(msg):
    logging.info(msg)

class DisBertMixIn:
    def setup_layers(self, config, layer_ipu):
        self.bert_embeddings = poptorch.BeginBlock(self.distilbert.embeddings,"Embedding",0)
        for index, layer in enumerate(self.distilbert.transformer.layer):
            ipu = layer_ipu[index]
            #if index != config.num_hidden_layers - 1:
            self.distilbert.transformer.layer[index].attention = MultiHeadSelfAttention(config)
            self.distilbert.transformer.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger(f"Encoder {index:<2} --> IPU {ipu}")
        self.pre_classifier  = poptorch.BeginBlock(self.pre_classifier , "Classifier", ipu_id=ipu)
        self.classifier  = poptorch.BeginBlock(self.classifier , "Classifier", ipu_id=ipu)
        self.dropout  = poptorch.BeginBlock(self.dropout , "Classifier", ipu_id=ipu)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Prevent word_embedding serialization when loading from pretrained so weights are loaded
        embedding_serialization = 1
        if kwargs.get("config"):
            embedding_serialization = kwargs["config"].embedding_serialization_factor
            kwargs["config"].embedding_serialization_factor = 1
        model = super().from_pretrained(*args, **kwargs)

        # Apply serialization afterwards
        if embedding_serialization > 1:
            model.bert.embeddings.word_embeddings = SerializedEmbedding(model.bert.embeddings.word_embeddings,
                                                                        embedding_serialization)
        return model

    def save_pretrained(self, *args, **kwargs):
        # Unwrap the SerializedEmbedding layer before saving then wrap again
        if self.config.embedding_serialization_factor > 1:
            serialized = self.bert.embedding.word_embeddings
            deserialized = nn.Embedding.from_pretrained(torch.stack([l.weight for l in serialized.split_embeddings]), padding_idx=0)
            self.bert.embeddings.word_embeddings = deserialized
            super().save_pretrained(*args, **kwargs)
            self.bert.embeddings.word_embeddings = serialized
        else:
            super().save_pretrained(*args, **kwargs)

class BertMixIn:
    def setup_layers(self, config, layer_ipu):
        self.bert_embeddings = poptorch.BeginBlock(self.bert.embeddings,"Embedding",0)
        for index, layer in enumerate(self.bert.encoder.layer):
            ipu = layer_ipu[index]
            if index != config.num_hidden_layers - 1:
                self.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
                logger(f"Encoder {index:<2} --> IPU {ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=ipu)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Prevent word_embedding serialization when loading from pretrained so weights are loaded
        embedding_serialization = 1
        if kwargs.get("config"):
            embedding_serialization = kwargs["config"].embedding_serialization_factor
            kwargs["config"].embedding_serialization_factor = 1
        model = super().from_pretrained(*args, **kwargs)

        # Apply serialization afterwards
        if embedding_serialization > 1:
            model.bert.embeddings.word_embeddings = SerializedEmbedding(model.bert.embeddings.word_embeddings,
                                                                        embedding_serialization)
        return model

    def save_pretrained(self, *args, **kwargs):
        # Unwrap the SerializedEmbedding layer before saving then wrap again
        if self.config.embedding_serialization_factor > 1:
            serialized = self.bert.embedding.word_embeddings
            deserialized = nn.Embedding.from_pretrained(torch.stack([l.weight for l in serialized.split_embeddings]), padding_idx=0)
            self.bert.embeddings.word_embeddings = deserialized
            super().save_pretrained(*args, **kwargs)
            self.bert.embeddings.word_embeddings = serialized
        else:
            super().save_pretrained(*args, **kwargs)

class BertSentenceMixIn:
    def setup_layers(self, config, layer_ipu):
        print("A", dir(self))
        self.bert_embeddings = poptorch.BeginBlock(self.embeddings,"Embedding",0)
        for index, layer in enumerate(self.encoder.layer):
            ipu = layer_ipu[index]
            if index != config.num_hidden_layers - 1:
                self.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
                logger(f"Encoder {index:<2} --> IPU {ipu}")
        #self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=ipu)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Prevent word_embedding serialization when loading from pretrained so weights are loaded
        embedding_serialization = 1
        if kwargs.get("config"):
            embedding_serialization = kwargs["config"].embedding_serialization_factor
            kwargs["config"].embedding_serialization_factor = 1
        model = super().from_pretrained(*args, **kwargs)

        # Apply serialization afterwards
        if embedding_serialization > 1:
            model.bert.embeddings.word_embeddings = SerializedEmbedding(model.bert.embeddings.word_embeddings,
                                                                        embedding_serialization)
        return model

    def save_pretrained(self, *args, **kwargs):
        # Unwrap the SerializedEmbedding layer before saving then wrap again
        if self.config.embedding_serialization_factor > 1:
            serialized = self.embedding.word_embeddings
            deserialized = nn.Embedding.from_pretrained(torch.stack([l.weight for l in serialized.split_embeddings]), padding_idx=0)
            self.embeddings.word_embeddings = deserialized
            super().save_pretrained(*args, **kwargs)
            self.embeddings.word_embeddings = serialized
        else:
            super().save_pretrained(*args, **kwargs)

#export POPLAR_ENGINE_OPTIONS={"autoReport.directory":"/profile/output","autoReport.outputDebugInfo":"true"}