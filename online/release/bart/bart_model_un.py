
from transformers import BartTokenizerFast, BartConfig
from transformers import BartForConditionalGeneration
from generic_inference_model import ModelWrapper, Config
import torch
import dataclasses
from ipu_options import get_options
import poptorch

import sys
import numpy as np
from typing import Optional, Tuple

from search_utils import greedy_search, beam_search

import bart_encoder 
import bart_decoder 


#bart_checkpoint = "facebook/bart-large-cnn"
bart_checkpoint = "ainize/bart-base-cnn"

bart_tokenizer = BartTokenizerFast.from_pretrained(bart_checkpoint)


wrapped_encoder_model = bart_encoder.create(bart_checkpoint)
wrapped_decoder_model = bart_decoder.create(bart_checkpoint)

base_text = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft)."
#base_text = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building."

#base_text = base_text + base_text + base_text
tokenizer_ids = bart_tokenizer.encode_plus(base_text, max_length=512, padding='max_length',return_tensors="pt")
#sys.exit(0)
input_ids = tokenizer_ids['input_ids']
attention_mask = tokenizer_ids['attention_mask']



encoder_result = wrapped_encoder_model(input_ids, attention_mask)
#print("A", encoder_result.shape, attention_mask.shape)
#sys.exit()

result_length = 64
output_length = 32
batch_size = 4

decoder_ids = torch.zeros(batch_size, output_length).to(torch.int64)
encoder_result = encoder_result.expand(batch_size,encoder_result.shape[1],encoder_result.shape[2])
attention_mask = attention_mask.expand(batch_size,attention_mask.shape[1])

dynamic_mask = torch.zeros(batch_size, output_length).to(torch.int64)

decoder_ids[:,0] = 2
dynamic_mask[:,0] = 1

beam_scores = torch.zeros(batch_size,)
greedy = False

for i in range(1,32):
    #print("BBB", decoder_ids[:,:i+1], dynamic_mask[:,:i+1])

    decoder_result = wrapped_decoder_model(decoder_ids, 
        attention_mask=dynamic_mask,
        encoder_attention_mask=attention_mask, 
        encoder_hidden_states=encoder_result)

    if greedy:
        for x in range(batch_size):
            next_index = greedy_search(decoder_ids[x,i-1], decoder_result[x,-1,:])
            decoder_ids[x,i] = next_index 
    else:
        decoder_ids, beam_scores = beam_search(i, decoder_ids, decoder_result, beam_scores, i == 1)
        
    dynamic_mask[:,i] = 1   


print("C", bart_tokenizer.decode(decoder_ids[0]))
 






