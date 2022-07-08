import os
from PIL import Image
from typing import Dict
from torch import LongTensor
import torch
torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())

from .min_dalle_base import MinDalleBase
from .models.dalle_bart_encoder_torch import DalleBartEncoderTorch
from .models.dalle_bart_decoder_torch import DalleBartDecoderTorch, DalleBartDecoderSplitTorch
from .models.vqgan_detokenizer import VQGanDetokenizer
import poptorch
from .ipu_options import get_ipu_options
import threading

class MinDalleTorch(MinDalleBase):
    def __init__(
        self, 
        is_mega: bool, 
        is_reusable: bool = True,
        token_count: int = 256
    ):
        print("initializing MinDalleTorch")
        super().__init__(is_mega)
        self.is_reusable = is_reusable
        self.token_count = token_count
    
        self.encoder_params_path = os.path.join(self.model_path, 'encoder.pt')
        self.decoder_params_path = os.path.join(self.model_path, 'decoder.pt')
        self.detoker_params_path = os.path.join('min-dalle_int','pretrained', 'vqgan', 'detoker.pt')

        self.enable_ipu_encoder = False
        self.enable_ipu_decoder = True


        #if is_reusable:
        self.init_encoder()
        self.init_decoder()
        self.init_detokenizer()

        x = threading.Thread(target=self.generate_image_tokens, args=('avocado chair', 0))
        x.start()


    def init_encoder(self):
        print("initializing DalleBartEncoderTorch")
        self.encoder = DalleBartEncoderTorch(
            layer_count = self.config['encoder_layers'],
            embed_count = self.config['d_model'],
            attention_head_count = self.config['encoder_attention_heads'],
            text_vocab_count = self.config['encoder_vocab_size'],
            text_token_count = self.config['max_text_length'],
            glu_embed_count = self.config['encoder_ffn_dim']
        )
        params = torch.load(self.encoder_params_path)
        self.encoder.load_state_dict(params, strict=False)
        del params
        ipu_options = get_ipu_options()
        if self.enable_ipu_encoder:
            self.encoder.half()
            self.ipu_encoder = poptorch.inferenceModel(self.encoder, ipu_options)


    def init_decoder(self):
        print("initializing DalleBartDecoderTorch")

        if self.enable_ipu_decoder:
            decoder_class = DalleBartDecoderSplitTorch
        else:
            decoder_class = DalleBartDecoderTorch

        self.decoder = decoder_class(
            image_vocab_size = self.config['image_vocab_size'],
            image_token_count = self.config['image_length'],
            sample_token_count = self.token_count,
            embed_count = self.config['d_model'],
            attention_head_count = self.config['decoder_attention_heads'],
            glu_embed_count = self.config['decoder_ffn_dim'],
            layer_count = self.config['decoder_layers'],
            batch_count = 2,
            start_token = self.config['decoder_start_token_id'],
            is_verbose = True
        )
        params = torch.load(self.decoder_params_path)
        
        if self.enable_ipu_decoder:
            self.decoder.update_weights(params)
        else:
            self.decoder.load_state_dict(params, strict=False)
        del params
       
    def init_detokenizer(self):
        print("initializing VQGanDetokenizer")
        self.detokenizer = VQGanDetokenizer()
        params = torch.load(self.detoker_params_path)
        self.detokenizer.load_state_dict(params)
        del params
            

    def generate_image_tokens(self, text: str, seed: int) -> LongTensor:
        text_tokens = self.tokenize_text(text)
        text_tokens = torch.tensor(text_tokens).to(torch.long)

        #if not self.is_reusable: self.init_encoder()

        if not self.enable_ipu_encoder:
            encoder_state = self.encoder(text_tokens)
        else:
            #print("P", text_tokens[0,:].reshape((1,64)).shape)
            encoder_state0 = self.ipu_encoder(text_tokens[0,:].reshape((1,64)))
            encoder_state1 = self.ipu_encoder(text_tokens[1,:].reshape((1,64)))
            #print("E", encoder_state0.shape, encoder_state1.shape)
            encoder_state = torch.cat([encoder_state0, encoder_state1])

        #encoder_state = self.ipu_encoder(text_tokens)
        if not self.is_reusable: del self.encoder

        if not self.is_reusable: self.init_decoder()
        print("sampling image tokens", encoder_state.shape)
        torch.manual_seed(seed)
        #if not self.ipu_decoder:
        image_tokens = self.decoder.forward(text_tokens, encoder_state)
        #else:
        #    print("Here")
        #    image_tokens = self.ipu_decoder.compile(text_tokens, encoder_state)
        #    print("HHH")
        #    image_tokens = self.ipu_decoder(text_tokens, encoder_state)
        #    print("There")
        if not self.is_reusable: del self.decoder
        return image_tokens
        
    def generate_image_serve(self, text: str, seed: int) :
        image_tokens = self.generate_image_tokens(text, seed)
        print("detokenizing image")
        image = self.detokenizer.forward(image_tokens).to(torch.uint8)
        return image.to('cpu').detach().numpy()

    def generate_image(self, text: str, seed: int) -> Image.Image:
        image_tokens = self.generate_image_tokens(text, seed)
        if not self.is_reusable: self.init_detokenizer()
        print("detokenizing image")
        image = self.detokenizer.forward(image_tokens).to(torch.uint8)
        if not self.is_reusable: del self.detokenizer
        image = Image.fromarray(image.to('cpu').detach().numpy())
        return image