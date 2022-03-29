

import pickle
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

with open('save.pik','rb') as fp:
    data = pickle.load(fp)

masked = data['masked_lm_positions']
data = data['data']

#print("B", len(masked))
#print("A", masked[0][511])


#print("A", len(data), len(data[0]), len(data[0][0]))
#print("B", data[0][0].shape)
#print("C", data[0][0].shape)
#print("D", data[0][0].shape)

#print("E", data[0][0][15][1])
#print("F", data[0][1][15][1])

index = 0
total_results = []
for x in range(len(data[0])): # Batch Index
    for y in range(len(data[x][0][2])): # Item Index
        sequence_result = []
        for z in range(len(data[x][0][y])): # 
            if masked[x][y][z] != 0:
                tokens = tokenizer.convert_ids_to_tokens(data[x][1][y][z])
                result = {'index':masked[x][y][z].item(),'tokens':tokens, 'logits':data[x][0][y][z].numpy()}
                sequence_result.append(result)
        total_results.append(sequence_result)

print(total_results[0])


#print("AA", data[0][0][0][3])
#print("A", total_results[1])

#for x in range(len(data[0][0])):
#    p = data[0][0][x]
#    for y in range(len(p)):
#        if masked[x][y] == 0:
#            break
        
        #r = torch.topk(p[y],5,sorted=True,largest=True)
        #print("R", r)

#print("A", masked[54], data[0][0][0][0].shape)

