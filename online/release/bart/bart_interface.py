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
#from model.optimized_gpt2_attn import OptimizedGPT2Attention_test, OptimizedGPT2Attention_nobuffer
from tokenizers import Tokenizer
from transformers import BartModel

from transformers import BartTokenizer, BartTokenizerFast
from dataclasses import dataclass

import bart_encoder
import bart_decoder

from search_utils import greedy_search, beam_search

@dataclass
class Options:
    model_path:str = 'ainize/bart-base-cnn'
    input_len:int = 512
    output_length:int = 32
    batch_size:int = 4


class BartInterfaceWrapper:
    def __init__(self, options:Options=Options()):
        self.options = options
        self.create_ipu(options)

    def _create_encoder(self, checkpoint):
        return bart_encoder.create(checkpoint)

    def _create_decoder(self, checkpoint):
        return bart_decoder.create(checkpoint)

    # TODO : Create multiple decoders per encoder for better performance
    def create_ipu(self, options):
        self.encoder = self._create_encoder(options.model_path)
        self.decoder = self._create_decoder(options.model_path)
        print("Bart Model Running")

    def run_data(self, input, callback):
        tic = time.time()
        input_ids = torch.from_numpy(input[0])
        attention_mask = torch.from_numpy(input[1])

        encoder_result = self.encoder(input_ids, attention_mask)

        batch_size = self.options.batch_size
        output_length = self.options.output_length

        decoder_ids = torch.zeros(batch_size, output_length).to(torch.int64)
        encoder_result = encoder_result.expand(batch_size,encoder_result.shape[1],encoder_result.shape[2])
        attention_mask = attention_mask.expand(batch_size,attention_mask.shape[1])

        dynamic_mask = torch.zeros(batch_size, output_length).to(torch.int64)

        decoder_ids[:,0] = 2
        dynamic_mask[:,0] = 1
        beam_scores = torch.zeros(batch_size,)

        #print("Running Decoder")
        for i in range(1,output_length):
            tic = time.time()
            decoder_result = self.decoder(decoder_ids, 
                attention_mask=dynamic_mask,
                encoder_attention_mask=attention_mask, 
                encoder_hidden_states=encoder_result)
            #print("D", time.time() - tic)
            tic = time.time()
            decoder_ids, beam_scores = beam_search(i, decoder_ids, decoder_result, beam_scores, i == 1)
            #print("BB", time.time() - tic)
            dynamic_mask[:,i] = 1
        #print("Finished Decoder")

        callback(decoder_ids) 


        

       


def main():
    text = """The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure
    in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel 
    Tower is the second tallest free-standing structure in France after the Millau Viaduct."""

    tokenizer = BartTokenizerFast.from_pretrained('ainize/bart-base-cnn')
    result = tokenizer.tokenize(text)
    wrapper = BartTritonWrapper()
    wrapper.run_data(result)



if __name__ == '__main__':
    main()
