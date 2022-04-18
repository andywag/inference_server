
import dataclasses
from ipu_options import get_options
import poptorch
import torch
from transformers import BartForConditionalGeneration
from generic_inference_model import ModelWrapper, Config

class BartPopEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return output.last_hidden_state
       
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        return model

def create(checkpoint):
    print("Loading", checkpoint)
    bart_model = BartForConditionalGeneration.from_pretrained(checkpoint)
    config = Config()

    bart_model.config.update(dataclasses.asdict(config))
    options = get_options(config)
    wrapped_encoder = BartPopEncoder(bart_model.get_encoder())
    wrapped_encoder_model = poptorch.inferenceModel(wrapped_encoder, options)
    #wrapped_encoder.eval()
    #wrapped_encoder_model.compile(torch.zeros(4,512,dtype=torch.int32),torch.zeros(4,512,dtype=torch.int32))

    return wrapped_encoder_model