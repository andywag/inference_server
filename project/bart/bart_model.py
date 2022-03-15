
from transformers import BartTokenizerFast, BartConfig
from transformers.models.bart.modeling_bart import BartDecoderLayer
from hug_bart import BartForConditionalGeneration
from generic_inference_model import ModelWrapper, Config
import torch
import dataclasses
from ipu_options import get_options
import poptorch

from blocks import SerializedEmbedding
import sys
import numpy as np
from typing import Optional, Tuple

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    result =  e_x / e_x.sum()
    return result

class BartPopDecoder(torch.nn.Module):
    def __init__(self, model, lm_head):
        super().__init__()
        self.model = model
        self.lm_head = lm_head#torch.nn.Linear(model.config.d_model, model.config.vocab_size, bias=False)
        #print("A", dir(self.model))
        #for x in range(len(self.model.layers)): 
        #    self.model.layers[x] = BartDecoderLayerWrapper(self.model.config)
            

    def forward(self, input_ids, encoder_attention_mask, encoder_hidden_states, past_indices=None, dynamic_mask=None):
        
        output = self.model(input_ids,
            encoder_attention_mask=encoder_attention_mask, 
            encoder_hidden_states=encoder_hidden_states,
            past_indices=past_indices,
            dynamic_mask=dynamic_mask)
    
        result = self.lm_head(output.last_hidden_state)

        return result#output.last_hidden_state 
       
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        return model

class BartPopEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        #inputs = {
        #    "input_ids": input_ids,
        #}
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return output.last_hidden_state
       
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        return model


bart_checkpoint = "facebook/bart-large-cnn"
bart_checkpoint = "ainize/bart-base-cnn"
bart_model = BartForConditionalGeneration.from_pretrained(bart_checkpoint)
lm_head = bart_model.lm_head

bart_tokenizer = BartTokenizerFast.from_pretrained(bart_checkpoint)
config = Config()

bart_model.config.update(dataclasses.asdict(config))
options = get_options(config)

wrapped_encoder = BartPopEncoder(bart_model.get_encoder())
#wrapped_encoder.half()
wrapped_encoder.eval()
wrapped_encoder_model = poptorch.inferenceModel(wrapped_encoder, options)

wrapped_decoder = BartPopDecoder(bart_model.get_decoder(), lm_head)
#wrapped_decoder.half()
wrapped_decoder.eval()
wrapped_decoder_model = poptorch.inferenceModel(wrapped_decoder, options)

base_text = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft)."
#base_text = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building."

#base_text = base_text + base_text + base_text
tokenizer_ids = bart_tokenizer.encode_plus(base_text, max_length=143, padding='max_length',return_tensors="pt")
#sys.exit(0)
input_ids = tokenizer_ids['input_ids']
attention_mask = tokenizer_ids['attention_mask']

#print("A", input_ids, attention_mask,input_ids.shape)
#sys.exit(0)
#results = bart_model.prepare_inputs_for_generation(input_ids, attention_mask)


encoder_result = wrapped_encoder_model(input_ids, attention_mask)
print("A", encoder_result.shape, attention_mask.shape)
#sys.exit()

result_length = 64
output_length = 32
batch_size = 8

decoder_ids = 2*torch.ones(batch_size, 1).to(torch.int64)
encoder_result = encoder_result.expand(batch_size,encoder_result.shape[1],encoder_result.shape[2])
attention_mask = attention_mask.expand(batch_size,attention_mask.shape[1])

dynamic_mask = torch.zeros(batch_size, output_length).to(torch.int64)

decoder_ids[0,0] = 2
dynamic_mask[0,:] = 1

past_indices = None
past_indices = torch.from_numpy(np.asarray([0]*8))
for i in range(2):
    decoder_result = wrapped_decoder_model(decoder_ids, 
        encoder_attention_mask=attention_mask, 
        encoder_hidden_states=encoder_result,
        past_indices=past_indices,
        dynamic_mask=dynamic_mask)
    logits,r_i = torch.topk(decoder_result[:,-1,:], batch_size)
    print("AA", logits, r_i)
    probs = softmax(logits.numpy())
    index = np.argmax(probs, axis=-1)
    for x in range(len(index)):
        decoder_ids[x,0] = r_i[x,index[x]]
    past_indices = torch.from_numpy(index)
    #print("AAA", past_indices)

print("A", decoder_ids)
print("C", bart_tokenizer.decode(decoder_ids[0]))
 






