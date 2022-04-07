

import pickle
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

with open('save.pik','rb') as fp:
    data = pickle.load(fp)

print(len(data['result']))
print(data['result'][4][0])

#masked = data['masked_lm_positions']
#data = data['data']

#print("B", len(masked))
#print("A", masked[0][511])


#print("A", len(data), len(data[0]), len(data[0][0]))
#print("B", data[0][0].shape)
#print("C", data[0][0].shape)
#print("D", data[0][0].shape)

#print("E", data[0][0][15][1])
#print("F", data[0][1][15][1])

