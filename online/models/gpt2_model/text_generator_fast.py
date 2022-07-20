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

import argparse
import logging
import os
import pdb
import time
from datetime import datetime

import numpy as np

import poptorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from optimized_gpt2_attn import OptimizedGPT2Attention
from tokenizers import Tokenizer
from transformers import BertTokenizer, BertTokenizerFast, GPT2Config, GPT2LMHeadModel, GPT2Model
from utils import _get_layer_ipu, str_to_bool

MODEL_CONFIG = {'gpt2': 'config/config.json', 'gpt2-medium': 'config/config_medium.json',
                'gpt2-large': 'config/config_large.json', 'gpt2-xl': 'config/config_xl.json'}


def set_args():
    """
    Sets up the arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-medium',
                        choices=('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'), help='model to use')
    parser.add_argument('--model-path', default='', type=str, required=False, help='path of model to load')
    parser.add_argument('--temperature', default=1.2, type=float, required=False, help='temperature')
    parser.add_argument('--repetition-penalty', default=2.0, type=float, required=False, help="repetition_penalty")
    parser.add_argument('--topk', default=8, type=int, required=False, help='topk to choice')
    parser.add_argument('--topp', default=0.6, type=float, required=False, help='top cumulative probability')

    parser.add_argument('--save-samples-path', default="sample/", type=str, required=False, help="path to save samples")
    parser.add_argument('--input-len', type=int, default=64, help='Input sequence length (default = 64)')
    parser.add_argument('--output-len', type=int, default=50, help='length of generated text')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size (default = 10)')
    parser.add_argument('--batches-per-step', type=int, default=1, help='device iterations (default = 1)')
    parser.add_argument("--single-ipu", type=str_to_bool, nargs="?", const=True, default=False,
                        help="single ipu or not ")
    parser.add_argument('--layers-per-ipu', type=int, default=3, nargs="+",
                        help='Number of decoder layers per pipeline stage.')
    parser.add_argument("--fp16", type=str_to_bool, nargs="?", const=True, default=False,
                        help="fp16")
    parser.add_argument("--poptorch_loop", type=str_to_bool, nargs="?", const=True, default=False,
                    help="using poptorch_loop to avoid too much streamcopy")
    return parser.parse_args()


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocab size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class PredictNextHead(torch.nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model

    def lm_head(self, hidden_states):
        return torch.matmul(hidden_states, self.model.wte.weight.T)

    def forward(self, context, index_tensor, hidden_states, index_one_hot, next_index_onehot, reverse_next_index_onehot):
        batch_size = context.shape[0]
        last_context_hidden_states = torch.matmul(index_one_hot, hidden_states).view(batch_size, -1)
        next_token_logits = self.lm_head(last_context_hidden_states)
        next_token_logits = next_token_logits / self.args.temperature
        (next_token_value, next_token) = torch.topk(next_token_logits, 1)
        tmp = context * reverse_next_index_onehot.to(torch.int32)
        next_context = torch.add(tmp, next_token * next_index_onehot)
        return next_context, index_tensor


class GTP2Wrapper(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.count = args.output_len
        self.args = args
        if args.model_path:
            self.model = GPT2Model.from_pretrained(args.model_path)
        elif args.model:
            model_config = MODEL_CONFIG[args.model]
            config = GPT2Config().from_json_file(model_config)
            self.model = GPT2Model(config=config)
        else:
            raise RuntimeError("Either args.model_path or args.model should be set.")
        self.head = PredictNextHead(self.args, self.model)
        self.optimize()
        if not args.single_ipu:
            self.sharding_mapping()

    def optimize(self):
        for layer in self.model.h:
            GPT2Attn = OptimizedGPT2Attention(self.model.config)
            GPT2Attn.load_state_dict(layer.attn.state_dict())
            layer.attn = GPT2Attn
            layer.mlp.act = nn.functional.gelu

    def forward(self, context, index_tensor):
        def body(context, index_tensor):
            index_one_hot = torch.nn.functional.one_hot(
                index_tensor, num_classes=self.args.input_len + self.args.output_len).to(torch.float)
            index_tensor = index_tensor + 1
            next_index_onehot = torch.nn.functional.one_hot(
                index_tensor, num_classes=self.args.input_len + self.args.output_len)
            reverse_next_index_onehot = torch.abs(1 - next_index_onehot)
            hidden_states = self.model(context, past_key_values=None, return_dict=False)
            next_context, index_tensor = self.head(
                context, index_tensor, hidden_states[0], index_one_hot, next_index_onehot, reverse_next_index_onehot)
            return next_context, index_tensor
        if self.args.poptorch_loop:
            context, index_tensor = poptorch.for_loop(self.count, body, [context, index_tensor])
            return context, index_tensor
        else:
            return body(context, index_tensor)

    def sharding_mapping(self):
        print("-------------------- Device Allocation --------------------")
        print("Embedding  --> IPU 0")
        self.model.wte = poptorch.BeginBlock(self.model.wte, "emb", ipu_id=0)

        layer_ipu = _get_layer_ipu(self.args.layers_per_ipu)
        for index, layer in enumerate(self.model.h):
            ipu = layer_ipu[index]
            self.model.h[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            print(f"Layer {index:<2} --> IPU {ipu}")
        self.head = poptorch.BeginBlock(self.head, ipu_id=0)


def main():
    args = set_args()
    if args.poptorch_loop and not args.single_ipu:
        raise("poptorch_loop did not support multi IPUs")

    model = GTP2Wrapper(args)
    if args.single_ipu:
        mem_prop = {'IPU0': 0.2}
    else:
        mem_prop = {'IPU0': 0.2, 'IPU1': 0.2}
    # Set poptorch options
    opts = poptorch.Options().deviceIterations(args.batches_per_step)
    opts.autoRoundNumIPUs(True)
    opts.setAvailableMemoryProportion(mem_prop)
    opts._Popart.set("saveInitializersToFile", "weights.bin")
    if not args.single_ipu:
        opts.setExecutionStrategy(poptorch.ShardedExecution())
    tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-poem")
    if args.fp16:
        model.half()
    model.eval()
    model = poptorch.inferenceModel(model, opts)
    text = "[CLS]George Washington was the first"
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids = torch.tensor(text_ids).long()
    token_len = len(input_ids)
    padding_size = args.input_len + args.output_len - token_len
    padding = torch.ones(args.batches_per_step * args.batch_size, padding_size) * (0)
    input_ids = torch.cat([input_ids.view(1, -1), padding], axis=-1).to(torch.int64)
    last_input_index = torch.tensor([1]).to(torch.int64) * (token_len - 1)
    model_time = []
    end_id = 102

    if args.poptorch_loop:
        ##compile#
        start_time = time.time()
        in1_ = input_ids.clone()
        in2_ = last_input_index.clone()
        _, _ = model(in1_, in2_)
        end_time = time.time()
        model_time.append(end_time - start_time)
        ##########
        start_time = time.time()
        input_ids, last_input_index = model(input_ids, last_input_index)
        end_time = time.time()
        model_time.append(end_time - start_time)
        print("Latency: {0} sec/sentence_({1} + {2})".format(np.mean(model_time[1:]),args.input_len, args.output_len)) 
        print("Latency: {} sec/token".format(np.mean(model_time[1:])/args.output_len))    
    else:
        for _ in range(args.output_len):
            start_time = time.time()
            input_ids, last_input_index = model(input_ids, last_input_index)
            end_time = time.time()
            model_time.append(end_time - start_time)
            print("Latency: {} sec/token".format(np.mean(model_time[1:])))
            if end_id in input_ids.view(-1).tolist():
                break

    print('compile times: {} s'.format(model_time[0]))
    print('total times: {} s'.format(sum(model_time[1:])))
    text = ''
    for id in input_ids.view(-1).tolist():
        text += tokenizer._convert_id_to_token(id)
    print(text)


if __name__ == '__main__':
    main()
