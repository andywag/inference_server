
import poptorch
import logging
logger = logging.getLogger()


def logger(msg):
    logging.info(msg)

class BertMixIn:
    def setup_layers(self, config, layer_ipu):
        self.bert_embeddings = poptorch.BeginBlock(self.bert.embeddings,"Embedding",0)
        for index, layer in enumerate(self.bert.encoder.layer):
            ipu = layer_ipu[index]
            if index != config.num_hidden_layers - 1:
                self.bert.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
                logger(f"Encoder {index:<2} --> IPU {ipu}")
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=ipu)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Prevent word_embedding serialization when loading from pretrained so weights are loaded
        embedding_serialization = 1
        if kwargs.get("config"):
            embedding_serialization = kwargs["config"].embedding_serialization_factor
            kwargs["config"].embedding_serialization_factor = 1
        model = super().from_pretrained(*args, **kwargs)

        # Apply serialization afterwards
        if embedding_serialization > 1:
            model.bert.embeddings.word_embeddings = SerializedEmbedding(model.bert.embeddings.word_embeddings,
                                                                        embedding_serialization)
        return model

    def save_pretrained(self, *args, **kwargs):
        # Unwrap the SerializedEmbedding layer before saving then wrap again
        if self.config.embedding_serialization_factor > 1:
            serialized = self.bert.embedding.word_embeddings
            deserialized = nn.Embedding.from_pretrained(torch.stack([l.weight for l in serialized.split_embeddings]), padding_idx=0)
            self.bert.embeddings.word_embeddings = deserialized
            super().save_pretrained(*args, **kwargs)
            self.bert.embeddings.word_embeddings = serialized
        else:
            super().save_pretrained(*args, **kwargs)

class BertSentenceMixIn:
    def setup_layers(self, config, layer_ipu):
        print("A", dir(self))
        self.bert_embeddings = poptorch.BeginBlock(self.embeddings,"Embedding",0)
        for index, layer in enumerate(self.encoder.layer):
            ipu = layer_ipu[index]
            if index != config.num_hidden_layers - 1:
                self.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
                logger(f"Encoder {index:<2} --> IPU {ipu}")
        #self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", ipu_id=ipu)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Prevent word_embedding serialization when loading from pretrained so weights are loaded
        embedding_serialization = 1
        if kwargs.get("config"):
            embedding_serialization = kwargs["config"].embedding_serialization_factor
            kwargs["config"].embedding_serialization_factor = 1
        model = super().from_pretrained(*args, **kwargs)

        # Apply serialization afterwards
        if embedding_serialization > 1:
            model.bert.embeddings.word_embeddings = SerializedEmbedding(model.bert.embeddings.word_embeddings,
                                                                        embedding_serialization)
        return model

    def save_pretrained(self, *args, **kwargs):
        # Unwrap the SerializedEmbedding layer before saving then wrap again
        if self.config.embedding_serialization_factor > 1:
            serialized = self.embedding.word_embeddings
            deserialized = nn.Embedding.from_pretrained(torch.stack([l.weight for l in serialized.split_embeddings]), padding_idx=0)
            self.embeddings.word_embeddings = deserialized
            super().save_pretrained(*args, **kwargs)
            self.embeddings.word_embeddings = serialized
        else:
            super().save_pretrained(*args, **kwargs)