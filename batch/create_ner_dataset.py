
from datasets import load_dataset
from transformers import AutoTokenizer
from adlfs import AzureBlobFileSystem

dataset = load_dataset("conll2003")
dataset = dataset["train"]
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

dataset = dataset.map(lambda x: {'tags':x['ner_tags']})
print("A", dataset.column_names)

fs = AzureBlobFileSystem(connection_string="DefaultEndpointsProtocol=https;AccountName=andynlpstore;AccountKey=hkMiWLiqIpONH0NnyhmYAO9SmdVJZb1CazjCB6mnk/72ee5KdyKnq/ByS5s6/ZPUPbP2HImIveIvwxYSP88Reg==;EndpointSuffix=core.windows.net")
dataset.save_to_disk('graphcore/connl_ner',fs=fs)
