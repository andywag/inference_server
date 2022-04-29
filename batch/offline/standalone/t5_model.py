
from transformers import T5ForConditionalGeneration, T5Config
import forge as f
import poptorch
import logging
def logger(msg):
    logging.info(msg)

def _get_layer_ipu(layers_per_ipu):
    # List of the IPU Id for each encoder layer
    layer_ipu = []
    for ipu, n_layers in enumerate(layers_per_ipu):
        layer_ipu += [ipu] * n_layers
    return layer_ipu

class T5Wrapper(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.setup_layers(config, _get_layer_ipu(config.layers_per_ipu))

    def setup_layers(self, config, layer_ipu):
        self.shared = poptorch.BeginBlock(self.shared,"Embedding",0)
        print("A", layer_ipu)
        for index, layer in enumerate(self.encoder.block):
            ipu = layer_ipu[index]
            if index != config.num_hidden_layers - 1:
                self.encoder.block[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
                logger(f"Encoder {index:<2} --> IPU {ipu}")

        print("Done B")
        for index, layer in enumerate(self.decoder.block):
            ipu = layer_ipu[index + len(self.encoder.block)]
            if index != config.num_hidden_layers - 1:
                self.decoder.block[index] = poptorch.BeginBlock(layer, f"Decoder{index}", ipu_id=ipu)
                logger(f"Decoder {index:<2} --> IPU {ipu}")

        self.lm_head = poptorch.BeginBlock(self.lm_head, "Classifier", ipu_id=ipu)

    
    @f.sign(f.self,f.arg('input_ids'),f.arg('attention_mask'),f.arg('labels'))
    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return output.loss, output.logits