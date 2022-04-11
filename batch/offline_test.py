

from adlfs import AzureBlobFileSystem
from cloud_utils import CloudFileContainer
from transformers import AutoTokenizer
import numpy as np

endpoint = "DefaultEndpointsProtocol=https;AccountName=andynlpstore;AccountKey=hkMiWLiqIpONH0NnyhmYAO9SmdVJZb1CazjCB6mnk/72ee5KdyKnq/ByS5s6/ZPUPbP2HImIveIvwxYSP88Reg==;EndpointSuffix=core.windows.net"
container = CloudFileContainer('AzureBlob', endpoint=endpoint)
dataset = container.load_dataset("graphcore/masked_small,text")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def create_position(input_ids, max_len):
    new_result = np.zeros(shape=(max_len,), dtype=np.int)
    result = np.where(np.asarray(input_ids) == 103)
    
    max_padding = min(max_len, len(result[0]))
    new_result[:max_padding] = result[0][:max_padding]
    return new_result

tokenized_dataset = dataset.map(lambda x: tokenizer(x['label_text'],
    max_length=32, truncation=True, pad_to_max_length=True),batched=True)
tokenized_dataset = tokenized_dataset.map(lambda batch: {"masked_lm_labels": batch["input_ids"]}, batched=True)

tokenized_dataset = tokenized_dataset.map(lambda x: tokenizer(x['text'],
    max_length=384, truncation=True, pad_to_max_length=True),batched=True)

tokenized_dataset = tokenized_dataset.map(lambda x: {'masked_lm_positions':create_position(x['input_ids'], 32)})

print(tokenized_dataset.column_names)
print(tokenized_dataset[0])
