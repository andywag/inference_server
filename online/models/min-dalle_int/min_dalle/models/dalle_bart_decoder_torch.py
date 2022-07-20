from typing import List, Tuple
import torch
from torch import LongTensor, nn, FloatTensor, BoolTensor
torch.set_grad_enabled(False)

from .dalle_bart_encoder_torch import GLUTorch, AttentionTorch
import poptorch
from ..ipu_options import get_ipu_options

class DecoderCrossAttentionTorch(AttentionTorch):
    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(decoder_state)
        return super().forward(keys, values, queries, attention_mask)


class DecoderSelfAttentionTorch(AttentionTorch):
    def forward(
        self, 
        decoder_state: FloatTensor,
        attention_state: FloatTensor,
        attention_mask: BoolTensor,
        token_mask: BoolTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        keys = self.k_proj.forward(decoder_state)
        values = self.v_proj.forward(decoder_state)
        queries = self.q_proj.forward(decoder_state)
        attention_state = torch.where(
            token_mask[None, :, None], 
            torch.cat([keys, values]), 
            attention_state
        )
        batch_count = decoder_state.shape[0]
        keys = attention_state[:batch_count]
        values = attention_state[batch_count:]
        decoder_state = super().forward(keys, values, queries, attention_mask)
        return decoder_state, attention_state


class DecoderLayerTorch(nn.Module):
    def __init__(
        self, 
        image_token_count: int,
        head_count: int, 
        embed_count: int,
        glu_embed_count: int
    ):
        super().__init__()
        self.image_token_count = image_token_count
        self.pre_self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.self_attn = DecoderSelfAttentionTorch(head_count, embed_count)
        self.self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.pre_encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.encoder_attn = DecoderCrossAttentionTorch(head_count, embed_count)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.glu = GLUTorch(embed_count, glu_embed_count)

        self.token_indices = torch.arange(self.image_token_count)
        if torch.cuda.is_available():
            self.token_indices = self.token_indices.cuda()

    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        attention_mask: BoolTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        # Self Attention
        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm.forward(decoder_state)
        self_attn_mask = self.token_indices < token_index + 1
        token_mask = self.token_indices == token_index
        self_attn_mask = torch.stack([self_attn_mask] * decoder_state.shape[0])
        decoder_state, attention_state = self.self_attn.forward(
            decoder_state,
            attention_state,
            self_attn_mask,
            token_mask
        )
        decoder_state = self.self_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Cross Attention
        residual = decoder_state
        decoder_state = self.pre_encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = self.encoder_attn.forward(
            decoder_state,
            encoder_state,
            attention_mask
        )
        decoder_state = self.encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # Feed forward
        residual = decoder_state
        decoder_state = self.glu.forward(decoder_state)
        decoder_state = residual + decoder_state

        return decoder_state, attention_state


class DalleBartDecoderBlockTorch(nn.Module):
    def __init__(
        self,
        image_vocab_size: int,
        image_token_count: int,
        sample_token_count: int,
        embed_count: int,
        attention_head_count: int,
        glu_embed_count: int,
        layer_count: int,
        batch_count: int,
        start_token: int,
        is_verbose: bool
    ):
        super().__init__()
        self.is_verbose = is_verbose
        self.layer_count = layer_count
        self.sample_token_count = sample_token_count
        self.condition_factor = 10.0
        self.image_token_count = image_token_count
        self.embed_tokens = nn.Embedding(image_vocab_size + 1, embed_count)
        self.embed_positions = nn.Embedding(image_token_count, embed_count)
        self.layers: List[DecoderLayerTorch] = nn.ModuleList([
            DecoderLayerTorch(
                image_token_count,
                attention_head_count,
                embed_count,
                glu_embed_count
            ) 
            for _ in range(layer_count)
        ])
        self.layernorm_embedding = nn.LayerNorm(embed_count)
        #self.layernorm_embedding = poptorch.autocast(enabled=True)(self.layernorm_embedding)
        self.final_ln = nn.LayerNorm(embed_count)
        self.lm_head = nn.Linear(embed_count, image_vocab_size + 1, bias=False)
        self.attention_state_shape = (
            layer_count,
            2 * batch_count,
            image_token_count,
            embed_count
        )
        self.zero_prob = torch.zeros([1])
        self.token_indices = torch.arange(self.sample_token_count)
        self.start_token = torch.tensor([start_token]).to(torch.long)
        if torch.cuda.is_available():
            self.zero_prob = self.zero_prob.cuda()
            self.token_indices = self.token_indices.cuda()
            self.start_token = self.start_token.cuda()
        
        #print("AAA", self.modules)
        for mod in self.modules():
            if isinstance(mod,  nn.LayerNorm):
                #print("DDD", mod)
                mod.forward = poptorch.autocast(enabled=True)(mod.forward)
            #if isinstance(mod,  nn.Embedding):
            #    mod.forward = poptorch.autocast(enabled=True)(mod.forward)
            #elif isinstance(mod,  nn.Linear):
            #    mod.forward = poptorch.autocast(enabled=True)(mod.forward)

    def forward(
        self,
        text_tokens: LongTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        prev_token: LongTensor,
        token_index: LongTensor
    ) -> Tuple[LongTensor, FloatTensor]:
        attention_mask = text_tokens.not_equal(1)
        batch_count = encoder_state.shape[0]
        prev_token_batched = torch.cat([prev_token] * batch_count)
        token_index_batched = torch.cat([token_index] * batch_count)
        decoder_state = self.embed_tokens.forward(prev_token_batched)
        decoder_state += self.embed_positions.forward(token_index_batched)
        decoder_state = self.layernorm_embedding.forward(decoder_state)
        decoder_state = decoder_state[:, None]
        attention_states_new = []
        for i in range(self.layer_count):
            decoder_state, attention_state_layer = self.layers[i].forward(
                decoder_state,
                encoder_state,
                attention_state[i],
                attention_mask,
                token_index
            )
            attention_states_new.append(attention_state_layer)
        decoder_state = self.final_ln(decoder_state)

        logits = self.lm_head(decoder_state)
        a = self.condition_factor
        logits: FloatTensor = (1 - a) * logits[0, -1] + a * logits[1, -1]

        top_logits, _ = logits.topk(8, dim=-1)
        probs = torch.where(
            logits < top_logits[-1],
            self.zero_prob,
            torch.exp(logits - top_logits[0])
        )
        return probs, torch.stack(attention_states_new), top_logits


class DalleBartDecoderSplitTorch():
    def __init__(
        self,
        image_vocab_size: int,
        image_token_count: int,
        sample_token_count: int,
        embed_count: int,
        attention_head_count: int,
        glu_embed_count: int,
        layer_count: int,
        batch_count: int,
        start_token: int,
        is_verbose: bool
    ):
        super().__init__()
        self.is_verbose = is_verbose
        self.layer_count = layer_count
        self.sample_token_count = sample_token_count
        self.condition_factor = 10.0
        self.image_token_count = image_token_count
       
        ipu_enable = True
       
        self.zero_prob = torch.zeros([1])
        self.token_indices = torch.arange(self.sample_token_count)
        self.start_token = torch.tensor([start_token]).to(torch.long)

        self.decoder = DalleBartDecoderBlockTorch(image_vocab_size,
            image_token_count,
            sample_token_count,
            embed_count,
            attention_head_count,
            glu_embed_count,
            layer_count,
            batch_count,
            start_token,
            is_verbose)

        ipu_options = get_ipu_options()
        if ipu_enable:
            self.decoder = self.decoder.half()
            self.decoder = poptorch.inferenceModel(self.decoder, ipu_options)

    def update_weights(self, params):
        #print(params)
        self.decoder.load_state_dict(params, strict=False)


    def forward(
        self,
        text_tokens: LongTensor,
        encoder_state: FloatTensor
    ) -> LongTensor:
        image_tokens: List[LongTensor] = []
        attention_state = torch.zeros(self.decoder.attention_state_shape).to(torch.half)
        image_token = self.start_token


        for i in range(self.sample_token_count):
        
            probs, attention_state, top_logits = self.decoder(
                text_tokens = text_tokens,
                encoder_state = encoder_state,
                attention_state = attention_state,
                prev_token = image_token,
                token_index = self.token_indices[[i]]
            )
            
            probs = probs.type(torch.FloatTensor)
            image_token = torch.multinomial(probs, 1)
            image_tokens += [image_token]
            
        return torch.cat(image_tokens)

   


class DalleBartDecoderTorch(nn.Module):
    def __init__(
        self,
        image_vocab_size: int,
        image_token_count: int,
        sample_token_count: int,
        embed_count: int,
        attention_head_count: int,
        glu_embed_count: int,
        layer_count: int,
        batch_count: int,
        start_token: int,
        is_verbose: bool
    ):
        super().__init__()
        self.is_verbose = is_verbose
        self.layer_count = layer_count
        self.sample_token_count = sample_token_count
        self.condition_factor = 10.0
        self.image_token_count = image_token_count
        self.embed_tokens = nn.Embedding(image_vocab_size + 1, embed_count)
        self.embed_positions = nn.Embedding(image_token_count, embed_count)
        self.layers: List[DecoderLayerTorch] = nn.ModuleList([
            DecoderLayerTorch(
                image_token_count,
                attention_head_count,
                embed_count,
                glu_embed_count
            ) 
            for _ in range(layer_count)
        ])
        self.layernorm_embedding = nn.LayerNorm(embed_count)
        self.final_ln = nn.LayerNorm(embed_count)
        self.lm_head = nn.Linear(embed_count, image_vocab_size + 1, bias=False)
        self.attention_state_shape = (
            layer_count,
            2 * batch_count,
            image_token_count,
            embed_count
        )
        self.zero_prob = torch.zeros([1])
        self.token_indices = torch.arange(self.sample_token_count)
        self.start_token = torch.tensor([start_token]).to(torch.long)
        

    def decode_step(
        self,
        text_tokens: LongTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        prev_token: LongTensor,
        token_index: LongTensor
    ) -> Tuple[LongTensor, FloatTensor]:
        attention_mask = text_tokens.not_equal(1)
        batch_count = encoder_state.shape[0]
        prev_token_batched = torch.cat([prev_token] * batch_count)
        token_index_batched = torch.cat([token_index] * batch_count)
        decoder_state = self.embed_tokens.forward(prev_token_batched)
        decoder_state += self.embed_positions.forward(token_index_batched)
        decoder_state = self.layernorm_embedding.forward(decoder_state)
        decoder_state = decoder_state[:, None]
        attention_states_new = []

        for i in range(self.layer_count):
            decoder_state, attention_state_layer = self.layers[i].forward(
                decoder_state,
                encoder_state,
                attention_state[i],
                attention_mask,
                token_index
            )
            attention_states_new.append(attention_state_layer)
        decoder_state = self.final_ln(decoder_state)
        logits = self.lm_head(decoder_state)
        a = self.condition_factor
        logits: FloatTensor = (1 - a) * logits[0, -1] + a * logits[1, -1]

        top_logits, _ = logits.topk(50, dim=-1)
        probs = torch.where(
            logits < top_logits[-1],
            self.zero_prob,
            torch.exp(logits - top_logits[0])
        )
        return probs, torch.stack(attention_states_new)


    def forward(
        self,
        text_tokens: LongTensor,
        encoder_state: FloatTensor
    ) -> LongTensor:
        image_tokens: List[LongTensor] = []
        attention_state = torch.zeros(self.attention_state_shape)
        image_token = self.start_token

        for i in range(self.sample_token_count):
            #print("B", encoder_state, attention_state[0,0,:], image_token, self.token_indices[[i]])

            probs, attention_state = self.decode_step(
                text_tokens = text_tokens,
                encoder_state = encoder_state,
                attention_state = attention_state,
                prev_token = image_token,
                token_index = self.token_indices[[i]]
            )
            #print("StepA" + str(i), attention_state[0,0,:])
            #if i == 0:
            #    import sys
            #    sys.exit(0)
            

            image_token = torch.multinomial(probs, 1)
            image_tokens += [image_token]
            
        return torch.cat(image_tokens)