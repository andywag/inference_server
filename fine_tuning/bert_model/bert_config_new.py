
from dataclasses import dataclass
from enum import Enum, unique
from fine_tune_config import ModelSpecific
from .modeling import PipelinedBertForSequenceClassification
import ctypes
import os

        

def handle_custom_ops(config):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    CUSTOM_OP_PATH = os.path.join(file_dir, "custom_ops.so")
    if os.path.exists(CUSTOM_OP_PATH):
        ops_and_patterns = ctypes.cdll.LoadLibrary(CUSTOM_OP_PATH)
        ops_and_patterns.setVocabSize(config.vocab_size)
        ops_and_patterns.setEmbeddingSize(config.hidden_size)
        ops_and_patterns.setHiddenSize(config.hidden_size)
    else:
        exit()


@dataclass
class BertSpecific(ModelSpecific):
    model_path:str="./bert_model"
    sequence_length:int=384
    embedding_serialization_factor:int=1
    tuning_type:str = "Sequence"
    num_labels:int = 3

    

@dataclass 
class BertDescription(ModelDescription):
    model_specific:BertSpecific

    def get_model(self, model_description, config, half=True):
        handle_custom_ops(config)
        config.embedding_serialization_factor=self.embedding_serialization_factor
        config.layers_per_ipu=model_description.ipu_layout.layers_per_ipu
        config.recompute_checkpoint_every_layer=model_description.ipu_options.recompute_checkpoint_every_layer
        config.num_labels = num_labels
        print("Creating", num_labels)

        if self.tuning_type == "Sequence":
            model = PipelinedBertForSequenceClassification.from_pretrained(model_description.checkpoint, config=config).half()
            return model
        elif self.tuning_type == "Token":
            model = PipelinedBertForTokenClassification.from_pretrained(model_description.checkpoint, config=config).half()
            return model
        else:
            print("Error")