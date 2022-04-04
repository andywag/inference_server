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
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import (GPT2Config,GPT2LMHeadModel, GPT2Model)

#from utils import _get_layer_ipu, str_to_bool
from transformers import GPT2Tokenizer, GPT2TokenizerFast
import poptorch
from model.optimized_gpt2_attn import OptimizedGPT2Attention_test, OptimizedGPT2Attention_nobuffer


MODEL_CONFIG = {'gpt2': 'config/config.json', 'gpt2-medium': 'config/config_medium.json',
                'gpt2-large': 'config/config_large.json', 'gpt2-xl': 'config/config_xl.json'}

from dataclasses import dataclass

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    result =  e_x / e_x.sum()
    return result

@dataclass
class Options:
    model_path:str = 'gpt2'
    topk:int = 8
    input_len:int = 128
    output_len:int = 512
    batch_size:int = 1
    batches_per_step:int = 1
    single_ipu:bool = True
    poptorch_loop:bool = False



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
        self.nop = poptorch.nop
        self.optimize()
        if not args.single_ipu:
            self.sharding_mapping()

    def optimize(self):
        self.model.config.batch = self.args.batch_size
        self.model.config.seq = self.args.input_len + self.args.output_len
        self.model.config.input_len = self.args.input_len
        self.model.config.output_len = self.args.output_len
        for layer in self.model.h:
            if self.args.poptorch_loop:
              GPT2Attn = OptimizedGPT2Attention_nobuffer(self.model.config)
            else:
              GPT2Attn = OptimizedGPT2Attention_test(self.model.config)
            GPT2Attn.load_state_dict(layer.attn.state_dict(), strict=False)
            layer.attn = GPT2Attn
            #layer.mlp.act = nn.functional.gelu

    def forward(self, context, dynamic_mask, position_ids):

        hidden_states = self.model(context, attention_mask=dynamic_mask,
                                    position_ids=position_ids, past_key_values=None, return_dict=False)
        hidden_states_ = self.nop(hidden_states[0])
        next_token_logits = torch.matmul(hidden_states_, self.model.wte.weight.T)
        #(next_token_value, next_token) = torch.topk(next_token_logits, 1)
        (next_token_value_a, next_token_a) = torch.topk(next_token_logits, 5)

        #####
        next_dynamic_mask = torch.cat(
            (torch.ones(self.args.batch_size, 1).to(torch.int64), dynamic_mask[:, :-1]), dim=-1)
        #####
        return next_dynamic_mask, (next_token_value_a, next_token_a)

    def sharding_mapping(self):
        print("-------------------- Device Allocation --------------------")
        print("Embedding  --> IPU 0")
        self.model.wte = poptorch.BeginBlock(self.model.wte, "emb", ipu_id=0)

        layer_ipu = _get_layer_ipu(self.args.layers_per_ipu)
        for index, layer in enumerate(self.model.h):
            ipu = layer_ipu[index]
            self.model.h[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            print(f"Layer {index:<2} --> IPU {ipu}")
        self.nop = poptorch.BeginBlock(self.nop, ipu_id=0)

class GPT2TritonWrapper:
    def __init__(self, options:Options=Options()):
        print("Creating GPT2 Wrapper")
        self.options = options
        self.model = self.create_ipu(options)

    def create_ipu(self, options):
        model = GTP2Wrapper(options)
        if options.single_ipu:
            mem_prop = {'IPU0': 0.2}
        else:
            mem_prop = {'IPU0': 0.2, 'IPU1': 0.2}
        # Set poptorch options
        opts = poptorch.Options().deviceIterations(options.batches_per_step)
        opts.autoRoundNumIPUs(True)
        opts.setAvailableMemoryProportion(mem_prop)
        opts._Popart.set("saveInitializersToFile", "weights.bin")
        if not options.single_ipu:
            opts.setExecutionStrategy(poptorch.ShardedExecution())
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        model.half()
        model.eval()
        model = poptorch.inferenceModel(model, opts)
        print("Finished Creating IPU")

        return model

    def run_data(self, input, callback):
        

        #print("Running Data")
        text_ids = list(input[0][0])
        input_length = int(input[1][0])
        output_length = int(input[2][0])
        #input_ids_all = torch.tensor(text_ids).long()
        #txt_len = len(text_ids)
        #all_ids = np.array([[text_ids[0]] for _ in range(self.options.batch_size)])
        input_ids = torch.ones(self.options.batch_size, 1).to(torch.int64)*text_ids.pop(0)

        #print("B", text_ids, input_length)

        position_ids = torch.zeros(self.options.batch_size, 1).to(torch.int64)
        dynamic_mask = torch.zeros(self.options.batch_size, self.options.input_len + self.options.output_len).to(torch.int64)
        dynamic_mask[:, 0] = 1
        end_id = 50256

        outputs = np.zeros((self.options.batch_size, self.options.output_len + self.options.input_len),np.uint32)

        for x in range(input_length + output_length):
            #print("Base", input_ids)
            outputs[:,x] = input_ids.view(self.options.batch_size, -1).numpy()
            dynamic_mask, result = self.model(input_ids, dynamic_mask, position_ids)
            
            if x < input_length-1:
                input_ids = torch.ones(self.options.batch_size, 1).to(torch.int64)*text_ids.pop(0)
            else:
                probs = softmax(result[0].numpy()).squeeze()

                input_ids = result[1][:,:,0]
                base = int(input_ids.squeeze().item())

                if base == outputs[0,x-1]:
                    select = 1
                elif probs[0] > .95:
                    select = 0
                else:
                    index = np.random.multinomial(1,probs)
                    select = 0
                    for x in range(len(index)):
                        if index[x] == 1:
                            select = x
                            break
                    #print("Probs", probs, index, select)

                    input_ids = result[1][:,:,select]
                    base = int(input_ids.squeeze().item())
                    if base == 0 or base == 198 or base == 50256:
                        break
                    #input_ids = input_ids_out

            position_ids += 1
            
    
        callback(outputs)


def main():
    args = set_args()
    if args.poptorch_loop and not args.single_ipu:
        raise("poptorch_loop did not support multi IPUs")

    model = self.create_ipu(args)

    dynamic_mask = torch.zeros(args.batch_size, args.input_len+args.output_len).to(torch.int64)
    dynamic_mask[:, 0] = 1
    text = "[CLS]My red dog went to the store to get some"
    # text =  "[CLS]床 前 明 月 光 ，"
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    input_ids_all = torch.tensor(text_ids).long()
    txt_len = len(text_ids)
    all_ids = np.array([[text_ids[0]] for _ in range(args.batch_size)])
    input_ids = torch.ones(args.batch_size, 1).to(torch.int64)*text_ids.pop(0)
    position_ids = torch.zeros(args.batch_size, 1).to(torch.int64)
    model_time = []
    end_id = 102
    if args.poptorch_loop:
        padding_size = args.input_len - txt_len
        padding = torch.ones(args.batch_size, padding_size) * (0)
        input_ids_all = input_ids_all.repeat(args.batch_size, 1)
        input_ids_all_pad = torch.concat([input_ids_all.view(args.batch_size, -1), padding], axis=-1).to(torch.int64)
        dynamic_mask[:, :txt_len+1] = 1
        position_ids += txt_len - 1
        ##compile##
        start_time = time.time()
        in1_ = input_ids_all_pad.clone()
        in2_ = dynamic_mask.clone()
        in3_ = position_ids.clone()
        _ = model(in1_, in2_, in3_)
        end_time = time.time()
        model_time.append(end_time - start_time)
        ############
        start_time = time.time()
        record = model(input_ids_all_pad, dynamic_mask, position_ids)
        end_time = time.time()
        model_time.append(end_time - start_time)
        output_tokens = torch.flip(record, dims=[1]).to(torch.int64)
        all_ids = torch.concat([input_ids_all.view(args.batch_size, -1).to(torch.int64), output_tokens], axis=-1)
        print("Latency: {0} sec/sentence_({1} + {2})".format(np.mean(model_time[1:]), args.input_len, args.output_len))
        print("Latency: {} sec/token".format(np.mean(model_time[1:])/(args.input_len + args.output_len)))
        print("Batch size: {0}; Input length {1}; Output length {2}, Throughput: {3} sentence/sec \n".format(args.batch_size, args.input_len, args.output_len, args.batch_size / np.mean(model_time[1:])))
    else:
        for _ in range(args.input_len + args.output_len):
            start_time = time.time()
            input_ids, dynamic_mask = model(input_ids, dynamic_mask, position_ids)
            end_time = time.time()
            model_time.append(end_time - start_time)
            position_ids += 1
            if len(text_ids) > 0:
                input_ids = torch.ones(args.batch_size, 1).to(torch.int64)*text_ids.pop(0)
            all_ids = np.concatenate((all_ids, input_ids.view(args.batch_size, -1).numpy()), axis=1)
            print("Latency: {} sec/token".format(np.mean(model_time[1:])))
            print("Batch size: {0}, throughput: {1} token/sec".format(args.batch_size,
                  args.batch_size / np.mean(model_time[1:])))

            finished_sample = 0
            for batch in range(args.batch_size):
                if end_id in all_ids[batch, :]:
                    finished_sample += 1
            if finished_sample >= args.batch_size:
                break
    text = ''
    for batch in all_ids.tolist():
        for id in batch:
            text += tokenizer._convert_id_to_token(id)
        text += '\n'
    text = text.replace("[SEP]", "")
    text = text.replace("[CLS]", "")
    print(text)


if __name__ == '__main__':
    main()
