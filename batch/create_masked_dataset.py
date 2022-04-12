from datasets import load_dataset, Dataset
from adlfs import AzureBlobFileSystem
import numpy as np
from transformers import AutoTokenizer, AutoConfig
import torch

wiki_data = load_dataset('wikitext','wikitext-103-v1')
wiki_data = wiki_data['train']

print("A", len(wiki_data))
print(wiki_data[3]['text'])

def mask_data(data):
    tokens = data.split(" ")
    masked = np.random.permutation(len(tokens))
    mask_length = int(.15*len(tokens))
    for x in range(mask_length):
        tokens[masked[x]] = "[MASK]"
    return " ".join(tokens)


result = []
for x in range(len(wiki_data)):
    if len(wiki_data[x]['text']) > 200:
        masked_data = mask_data(wiki_data[x]['text'])
        result.append(masked_data)

masked_dict = {'text':result}
masked_dataset = Dataset.from_dict(masked_dict)

print(len(masked_dataset))
print("A", masked_dataset[2])

fs = AzureBlobFileSystem(connection_string="DefaultEndpointsProtocol=https;AccountName=andynlpstore;AccountKey=hkMiWLiqIpONH0NnyhmYAO9SmdVJZb1CazjCB6mnk/72ee5KdyKnq/ByS5s6/ZPUPbP2HImIveIvwxYSP88Reg==;EndpointSuffix=core.windows.net")
result = fs.ls("")
#masked_dataset.save_to_disk('graphcore/masked_test',fs=fs)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenized_dataset = masked_dataset.map(lambda x: tokenizer(x['text'],
        max_length=384, truncation=True, pad_to_max_length=True,return_offsets_mapping=True),batched=True)
tokenized_dataset.set_format(type='torch')

position_values = (tokenized_dataset['input_ids'] == 103)
positions = position_values.nonzero()

masked_lm_positions=torch.zeros(size=(len(tokenized_dataset),32),dtype=torch.int)

row_index = [0]*len(positions)
for x in range(len(positions)):
    row = positions[x][0]
    col = row_index[row]
    row_index[row] += 1
    value = positions[x][1]
    if row_index[row] < 32:
        masked_lm_positions[row][col] = value
    

#for x in range(len(tokenized_dataset)):
#    index = 0
#    for y in range(384):
#        if tokenized_dataset['input_ids'][x][y] == 102:
#            break
#        elif tokenized_dataset['input_ids'][x][y] == 103:
#            masked_lm_positions[x][index] = x
#            index += 1
#        if index == 32:
#            break






