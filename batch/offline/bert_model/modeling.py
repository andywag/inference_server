# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
import poptorch
import transformers

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

from .bert_fused_attention import BertFusedSelfAttention
import os
import ctypes

from .bert_gc_mixin import BertMixIn

import forge as f

def logger(msg):
    logging.info(msg)


def handle_custom_ops(config):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    CUSTOM_OP_PATH = os.path.join(file_dir, "custom_ops.so")
    if os.path.exists(CUSTOM_OP_PATH):
        ops_and_patterns = ctypes.cdll.LoadLibrary(CUSTOM_OP_PATH)
        ops_and_patterns.setVocabSize(config.vocab_size)
        ops_and_patterns.setEmbeddingSize(config.hidden_size)
        ops_and_patterns.setHiddenSize(config.hidden_size)
    else:
        exit()

class OnehotGather(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_half = False

    def half(self):
        super().half()
        # Tracing is always executed in float as there are missing
        # implementations of operations in half on the CPU.
        # So we cannot query the inputs to know if we are running
        # with a model that has had .half() called on it.
        # To work around it nn.Module::half is overridden
        self._is_half = True

    def forward(self, sequence, positions):
        """
        Gather the vectors at the specific positions over a batch.
        """
        num_classes = int(sequence.shape[1])
        one_hot_positions = F.one_hot(positions, num_classes)
        if self._is_half:
            one_hot_positions = one_hot_positions.half()
        else:
            one_hot_positions = one_hot_positions.float()
        return torch.matmul(one_hot_positions.detach(), sequence)


class SerializedLinear(nn.Linear):
    def __init__(self, in_features, out_features, factor, bias=False,
                 mode=poptorch.MatMulSerializationMode.OutputChannels):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.factor = factor

    def forward(self, x):
        output = poptorch.serializedMatMul(x, self.weight.t(), self.mode, self.factor)
        if self.bias is not None:
            output += self.bias
        return output


def _get_layer_ipu(layers_per_ipu):
    # List of the IPU Id for each encoder layer
    layer_ipu = []
    for ipu, n_layers in enumerate(layers_per_ipu):
        layer_ipu += [ipu] * n_layers
    return layer_ipu


def recomputation_checkpoint(module: nn.Module):

    if not isinstance(module, nn.Module):
        raise RuntimeError("module is not an instance of torch.nn.Module.")

    class RecomputationCheckpointModule(type(module)):
        def forward(self, *x):
            return tuple(poptorch.recomputationCheckpoint(y) for y in super().forward(*x))

    if str(module.__class__) == str(RecomputationCheckpointModule):
        raise RuntimeError("module has already been assigned to a recomputation checkpoint.")

    RecomputationCheckpointModule.__name__ = type(module).__name__
    module.__class__ = RecomputationCheckpointModule

    return module


def accuracy(out, targ):
    return (out.argmax(dim=-1) == targ).float().mean()


def accuracy_masked(out, targ, mask_val):
    mask = (targ != mask_val).float()
    num_unmasked = mask.sum(1).unsqueeze(1)
    return (out.argmax(dim=-1) == targ).float().mul(mask).div(num_unmasked).sum(1).mean()


class PipelinedBertForPretraining(transformers.BertForMaskedLM, BertMixIn):
    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

        for layer in self.bert.encoder.layer:
            layer.attention.self = BertFusedSelfAttention(self.config)


        if self.config.embedding_serialization_factor > 1:
            self.cls.predictions.decoder = SerializedLinear(self.config.hidden_size,
                                                            self.config.vocab_size,
                                                            self.config.embedding_serialization_factor,
                                                            mode=poptorch.MatMulSerializationMode.OutputChannels)
            self.tie_weights()

        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)

        logger("-------------------- Device Allocation --------------------")
        logger("Embedding  --> IPU 0")
        self.bert.embeddings = poptorch.BeginBlock(self.bert.embeddings, "Embedding", ipu_id=0)

        for index, layer in enumerate(self.bert.encoder.layer):
            ipu = layer_ipu[index]
            layer = recomputation_checkpoint(layer) if self.config.recompute_checkpoint_every_layer else layer
            self.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger(f"Encoder {index:<2} --> IPU {ipu}")

        logger("Classifier --> IPU 0")
        self.cls = poptorch.BeginBlock(self.cls, "Classifier", ipu_id=0)
        logger("-----------------------------------------------------------")

    def _init_weights(self, module):
        """Initialize the weights"""
        def truncated_normal_(tensor, mean=0, std=1):
            """
            Truncated Normal distribution, truncated at 2 sigma
            """
            r = torch.tensor(truncnorm.rvs(-2, 2, loc=mean, scale=std, size=tensor.shape))
            tensor.data.copy_(r)

        if isinstance(module, nn.Linear):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @f.sign(f.self, f.arg('input_ids'), f.arg('attention_mask'), f.arg('token_type_ids'), f.arg('masked_lm_positions', default=None),f.arg('masked_lm_labels', default=None))
    def forward(self, **kwargs):
        
        masked_lm_positions = kwargs['masked_lm_positions']
        masked_lm_labels = kwargs['masked_lm_labels']
        del kwargs['masked_lm_positions']
        del kwargs['masked_lm_labels']

        outputs = self.bert(**kwargs)
        sequence_output = outputs[0]

        # Select only the masked tokens for the classifier
        masked_output = self.gather_indices(sequence_output, masked_lm_positions)
        
        prediction_scores = self.cls(masked_output)
        

        scores = prediction_scores
        #prediction_scores = self.cls(masked_output)
        if self.training:
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
                ignore_index=0).float()
            masked_lm_acc = accuracy_masked(prediction_scores.view([-1, masked_lm_labels.shape[1], self.config.vocab_size]), masked_lm_labels, 0)
            return masked_lm_loss, masked_lm_acc
        else:
            #
            masked_lm_acc = None
            
            if masked_lm_labels is not None:
                scores = torch.topk(prediction_scores, 5, axis=-1)
                masked_lm_acc = accuracy_masked(prediction_scores.view([-1, masked_lm_labels.shape[1], self.config.vocab_size]), masked_lm_labels, 0)
            #    print("B", masked_lm_acc)
            return scores, masked_lm_acc
       



class SerializedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, serialization_factor: int):
        super().__init__()
        self.serialization_factor = serialization_factor
        self.num_embeddings = embedding.num_embeddings

        # Num embeddings should be divisible by the serialization factor
        assert self.num_embeddings % self.serialization_factor == 0
        self.split_size = self.num_embeddings // self.serialization_factor
        self.split_embeddings = nn.ModuleList(
            [nn.Embedding.from_pretrained(embedding.weight[i*self.split_size:(i+1)*self.split_size, :].detach(),
                                          freeze=False,
                                          padding_idx=embedding.padding_idx if i == 0 else None)
             for i in range(self.serialization_factor)])

    def forward(self, indices):
        # iterate through the splits
        x_sum = None
        for i in range(self.serialization_factor):
            # mask out the indices not in this split
            split_indices = indices - i * self.split_size
            mask = (split_indices >= 0) * (split_indices < self.split_size)
            mask = mask.detach()
            split_indices *= mask

            # do the embedding lookup
            x = self.split_embeddings[i](split_indices)

            # multiply the output by mask
            x *= mask.unsqueeze(-1)

            # add to partial
            if x_sum is not None:
                x_sum += x
            else:
                x_sum = x
        return x_sum


class PipelinedBertForSequenceClassification(transformers.BertForSequenceClassification, BertMixIn):
    def __init__(self, config):
        super().__init__(config)
        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)
        self.setup_layers(self.config, layer_ipu)

    @f.sign(f.self,f.arg('input_ids'),f.arg('attention_mask'),f.arg('token_type_ids'),f.arg('labels', default=None))
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
       
        if self.training:
            final_loss = poptorch.identity_loss(output.loss, reduction="none")
            return final_loss, output.logits
        else:
            indices = torch.argmax(output.logits,dim=-1)
            return output.logits, indices

    

class PipelinedBertForTokenClassification(transformers.BertForTokenClassification, BertMixIn):
    def __init__(self, config):
        super().__init__(config)
        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)
        self.setup_layers(self.config, layer_ipu)

       
    @f.sign(f.self,f.arg('input_ids'),f.arg('attention_mask'),f.arg('token_type_ids'),f.arg('labels', default=None))
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        if self.training:
            final_loss = poptorch.identity_loss(output.loss, reduction="none")
            return final_loss, output.logits
        else:
            indices = torch.argmax(output.logits,dim=-1)
            return output.logits, indices

   


class PipelinedBertForQuestionAnswering(transformers.BertForQuestionAnswering, BertMixIn):
    def __init__(self, config):
        super().__init__(config)
        layer_ipu = _get_layer_ipu(self.config.layers_per_ipu)
        self.setup_layers(self.config, layer_ipu)

    @f.sign(f.self,f.arg('input_ids'),f.arg('attention_mask'),f.arg('token_type_ids'),f.arg('start_positions', default=None),f.arg('end_positions', default=None))
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        if self.training:
            final_loss = poptorch.identity_loss(output.loss, reduction="none")
            return final_loss, output.start_logits, output.end_logits
        else:
            return output.start_logits, output.end_logits

   
