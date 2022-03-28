from datasets import load_dataset, Dataset
from adlfs import AzureBlobFileSystem
import numpy as np

wiki_data = load_dataset('wikitext','wikitext-103-v1')
wiki_data = wiki_data['test']

print("A", len(wiki_data))
print(wiki_data[3]['text'])

def mask_data(data):
    tokens = data.split(" ")
    masked = np.random.permutation(len(tokens))
    mask_length = int(.15*len(tokens))
    #print(tokens, masked)
    for x in range(mask_length):
        tokens[masked[x]] = "[MASK]"
    #tokens[masked[:mask_length]] = "[MASK]"
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
masked_dataset.save_to_disk('graphcore/masked_test',fs=fs)


print(result)