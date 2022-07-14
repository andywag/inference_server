import dataclasses
from ipu_options import get_options
import poptorch
import torch
from transformers import BartForConditionalGeneration
from generic_inference_model import Config

class BartPopDecoder(torch.nn.Module):
    def __init__(self, model, lm_head):
        super().__init__()
        self.model = model
        self.lm_head = lm_head
        
    def forward(self, input_ids, attention_mask, encoder_attention_mask, encoder_hidden_states):
        
        output = self.model(input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask, 
            encoder_hidden_states=encoder_hidden_states,
            use_cache=None)
    
        result = self.lm_head(output.last_hidden_state)

        return result
       
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        return model

def create(checkpoint):
    bart_model = BartForConditionalGeneration.from_pretrained(checkpoint)
    config = Config()

    bart_model.config.update(dataclasses.asdict(config))
    options = get_options(config)
    wrapped_decoder = BartPopDecoder(bart_model.get_decoder(), bart_model.lm_head)
    wrapped_decoder.eval()
    wrapped_decoder_model = poptorch.inferenceModel(wrapped_decoder, options)
    return wrapped_decoder_model