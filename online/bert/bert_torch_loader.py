# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import popart
import json
from logging import getLogger

from bert_model import BertConfig, Bert

import pickle

logger = getLogger(__name__)


def get_mapping(config, task=None):
    init = {}
    embedding_proj = {
        "bert.embeddings.word_embeddings.weight": "Embedding/Embedding_Dict",
        "bert.embeddings.position_embeddings.weight": "Embedding/Positional_Dict",
        "bert.embeddings.token_type_embeddings.weight": "Embedding/Segment_Dict",
        "bert.embeddings.LayerNorm.weight": "Embedding/Gamma",
        "bert.embeddings.LayerNorm.bias": "Embedding/Beta",
    }
    init.update(**embedding_proj)
    for i in range(config.num_layers):
        if False:
            init.update(**{
                f"bert.encoder.layer.{i}.attention.self.query.weight": f"Layer{i}/Attention/Q",
                f"bert.encoder.layer.{i}.attention.self.key.weight": f"Layer{i}/Attention/K",
                f"bert.encoder.layer.{i}.attention.self.value.weight": f"Layer{i}/Attention/V",
            })
        else:
            init.update(**{
                f"bert.encoder.layer.{i}.attention.self.query.weight": f"Layer{i}/Attention/QKV",
                f"bert.encoder.layer.{i}.attention.self.key.weight": f"Layer{i}/Attention/QKV",
                f"bert.encoder.layer.{i}.attention.self.value.weight": f"Layer{i}/Attention/QKV",
                f"bert.encoder.layer.{i}.attention.self.query.bias": f"Layer{i}/Attention/QKV_Bias",
                f"bert.encoder.layer.{i}.attention.self.key.bias": f"Layer{i}/Attention/QKV_Bias",
                f"bert.encoder.layer.{i}.attention.self.value.bias": f"Layer{i}/Attention/QKV_Bias",
            })
           
        init.update(**{
            f"bert.encoder.layer.{i}.attention.output.dense.weight": f"Layer{i}/Attention/Out",
            f"bert.encoder.layer.{i}.attention.output.dense.bias": f"Layer{i}/Attention/Out_Bias",
            f"bert.encoder.layer.{i}.attention.output.LayerNorm.weight": f"Layer{i}/Attention/Gamma",
            f"bert.encoder.layer.{i}.attention.output.LayerNorm.bias": f"Layer{i}/Attention/Beta",
            f"bert.encoder.layer.{i}.intermediate.dense.weight": f"Layer{i}/FF/1/W",
            f"bert.encoder.layer.{i}.intermediate.dense.bias": f"Layer{i}/FF/1/B",
            f"bert.encoder.layer.{i}.output.dense.weight": f"Layer{i}/FF/2/W",
            f"bert.encoder.layer.{i}.output.dense.bias": f"Layer{i}/FF/2/B",
            f"bert.encoder.layer.{i}.output.LayerNorm.weight": f"Layer{i}/FF/Gamma",
            f"bert.encoder.layer.{i}.output.LayerNorm.bias": f"Layer{i}/FF/Beta",
        })
    if task is None or task=='SQUAD':
        squad_proj = {
            "qa_outputs.weight": "Squad/SquadW",
            "qa_outputs.bias": "Squad/SquadB",
        }
        init.update(**squad_proj)
    elif task == 'NER':
        squad_proj = {
            "classifier.weight": "Squad/NerW",
            "classifier.bias": "Squad/NerB",
        }
        init.update(**squad_proj)

    return init


def get_transform(config, task=None):
    init = {}

    def q_transform(arr):
        return arr[:, 0:config.hidden_size].T

    def k_transform(arr):
        return arr[:, config.hidden_size:config.hidden_size * 2].T

    def v_transform(arr):
        return arr[:, config.hidden_size * 2:config.hidden_size * 3].T

    for i in range(config.num_layers):
        layer = {
            #f"bert.encoder.layer.{i}.attention.self.query.weight": np.transpose,
            #f"bert.encoder.layer.{i}.attention.self.key.weight": np.transpose,
            #f"bert.encoder.layer.{i}.attention.self.value.weight": np.transpose,
            f"bert.encoder.layer.{i}.attention.output.dense.weight": np.transpose,
            f"bert.encoder.layer.{i}.intermediate.dense.weight": np.transpose,
            f"bert.encoder.layer.{i}.output.dense.weight": np.transpose,
        }
        init.update(**layer)
    if task is None or task == 'SQUAD':
        base = {
            "qa_outputs.weight": np.transpose
        }
        init.update(**base)
    else:
        squad_proj = {
            "classifier.weight": np.transpose
        }
        init.update(**squad_proj)
    return init

def generate_initializers(config, weights, mapping, transform={}, inference=True):
    """
    Generate a graph initializer dictionary from the tensor names and
    data loaded from either a checkpoint or frozen graph using one of
    the methods below (`load_tf_ckpt_data` or `load_tf_frozen_data`).

    In the general case, this will simply map the tensor names from the
    TF model to the Popart model.

    The exception is the query-key-value matrix which is formed by
    concatenating the weight tensors Q, K and V.
    """
    initializers = {}
    qkv_tensor_range = {
        "query": (0, config.hidden_size),
        "key": (config.hidden_size, config.hidden_size * 2),
        "value": (config.hidden_size * 2, config.hidden_size * 3),
    }


    for name, a in weights.items():
        array = a.numpy()
        #if "attention" in name and "bias" in name:
        #    continue
        if "pooler" in name or 'position_ids' in name:
            continue

        #logger.info(f"DATA: {name}, {np.var(array)}")
        #continue

        #logger.info(f"{name} -> {mapping[name]}, {np.mean(array)} {np.var(array)}")

        if "query" in name and inference:
            array = array/8.0

        if array.dtype == np.float32 and config.dtype == np.float16:
            array = array.astype(config.dtype)

        if name in transform:
            #logger.info(f"Transform1 {name} {array.shape}")
            array = transform[name](array)
            #logger.info(f"Transform2 {name} {array.shape}")

        # If it's part of QKV, we need to handle separately as those 3
        # tensors need concatenating into one
        is_qkv = mapping[name][-3:] == "QKV"
        is_qkv_bias = mapping[name][-8:-5] == "QKV"
        if is_qkv or is_qkv_bias:
            qkv_part = name.split(".")[-2]

            if mapping[name] not in initializers.keys():
                if is_qkv:
                    qkv_shape = (array.shape[0], array.shape[1] * 3)
                elif is_qkv_bias:
                    qkv_shape = (array.shape[0] * 3)
                initializers[mapping[name]] = np.empty(
                    qkv_shape, dtype=array.dtype
                )

            start_idx = qkv_tensor_range[qkv_part][0]
            end_idx = qkv_tensor_range[qkv_part][1]
            if is_qkv:
                initializers[mapping[name]][:, start_idx:end_idx] = array.T
            elif is_qkv_bias:
                initializers[mapping[name]][start_idx:end_idx] = array
                #print("Here", name, np.mean(array), start_idx, end_idx)
            logger.debug(f"Initialising QKV component {name}[{start_idx}:{end_idx}] from checkpoint")
            continue

        

        if mapping[name] == "Embedding/Embedding_Dict":
            tf_vocab_length = array.shape[0]
            diff = config.vocab_length - tf_vocab_length
            # Pad or Crop the vocab.
            if diff > 0:
                #logger.debug(f"Padding the vocabulary. From {tf_vocab_length} to {config.vocab_length}")
                pad = np.zeros((diff, config.hidden_size)).astype(array.dtype)
                array = np.concatenate((array, pad), axis=0)
            else:
                #logger.warning(f"Cropping the vocabulary may negatively effect performance. From {tf_vocab_length} to {config.vocab_length}")
                array = np.array(array[:config.vocab_length, :])

        # FIXME: This copy is currently required to prevent popart misinterpreting the memory layout after the transpose.
        # Remove once T13187 is resolved.
        initializers[mapping[name]] = array.copy()
    #for k,v in initializers.items():
    #    print("Init", k, np.mean(v), np.var(v))


    return initializers





def load_torch_data(path='converter/ner_large_model.bin'):
    import torch
    weights = torch.load(path,map_location=torch.device('cpu'))
    return weights


def create_checkpoint(fname, url):
    import requests
    print("Downloading Checkpoing", fname)
    r = requests.get(url)
    open(fname , 'wb').write(r.content)

def load_initializers_from_torch(file_link, file_name, config, task):
    """
    Loads weights, etc. from Tensorflow files into a dictionary of Numpy Arrays.

    Can read either checkpoint files, or frozen graphs, according to the
    `is_checkpoint` flag, passed in as the second argument.
    """
    import os
    import sys
    build_path = os.getenv("INFERENCE_BUILD")
    if build_path is None:
        print("Please Set INFERENCE_BUILD environment variable")
        sys.exit(0)

    file_path = f"{build_path}/pytorch_checkpoints"
    full_name = f"{file_path}/{file_name}"

    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if not os.path.exists(full_name):
        create_checkpoint(full_name, file_link)

    mapping = get_mapping(config, task=task)
    transform = get_transform(config, task=task)

    print("Loading Checkpoint", full_name)
    weights = load_torch_data(full_name)
    
    return generate_initializers(config, weights, mapping, transform)




