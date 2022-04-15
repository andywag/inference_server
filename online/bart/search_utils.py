

import numpy as np
import torch

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    result =  e_x / e_x.sum()
    return result

def greedy_search(last_input, decoder_result):
    logits,r_i = torch.topk(decoder_result, 2)
    probs = softmax(logits.numpy())
    index = np.argmax(probs, axis=-1)
    print("D", last_input, r_i)
    if last_input == r_i[0]:
        next_index = r_i[1]
    else:
        next_index = r_i[0]

    return next_index

def internal_search(scores):
    xd = len(scores)
    yd = len(scores[0])

    for x in xd:
        for y in yd:
            pass

def beam_search(index, decoder_inputs, decoder_result, beams, first=False):
    #print("Result", index, decoder_result.shape)
    scores = torch.nn.functional.log_softmax(decoder_result[:,index-1,:], dim=-1)

    logits,r_i = torch.topk(scores, len(beams))
    #probs = softmax(logits.numpy())
    #index = np.argmax(logits, axis=-1)
    #print("First", decoder_result[:,index-1,:4], scores[:,:4])
    #print("Second", decoder_result[0,:,0])
    if first:
        for x in range(len(beams)):
            beams[x] = logits[0,x]
            decoder_inputs[x,index] = r_i[0,x]
        #print('Returning', decoder_inputs[:,:index+1], logits)
        return decoder_inputs, logits[0]
    else:
        bl, bi = torch.topk(logits, len(beams))

        base_logits = logits
        #print("ACBD", base_logits, logits, beams)
        for x in range(len(beams)):
            for y in range(len(beams)):
                base_logits[x][y] = logits[x][y] + beams[x]
        #print("A", logits)
        new_logits = (base_logits).reshape(shape=(int(len(beams)**2),))
        base_value,base_index = torch.sort(new_logits,descending=True)
        #print("Base", base_value, base_index,r_i)

        selected = set()
        ind = 0
        for x in range(len(beams)*len(beams)):
            batch_index = int(base_index[x]/len(beams))
            select_index = base_index[x] % int(len(beams))

            if True or select_index not in selected:
                selected.add(select_index)
                #print("C", batch_index, select_index, index, r_i[batch_index, select_index])
                decoder_inputs[ind] = decoder_inputs[batch_index]
                decoder_inputs[ind][index] = r_i[batch_index, select_index]
                beams[ind] = new_logits[x]
                ind += 1
            if ind == 4:
                break
        #print('Return', decoder_inputs[:,:index+1], beams)
        return decoder_inputs, beams



        nl, ni = torch.topk(new_logits, len(beams))
        print("Sorted", a,b)

        #print("BBBB", nl, ni)

        #max_indices = np.argmax(new_logits, axis=-1)
        print("AAA", new_logits, r_i, nl, ni, bl, bi)

        for x in range(len(beams)):
            select_index = int(ni[x]/len(beams))
            batch_index = int(ni[x]) % int(len(beams))
            print("C", batch_index, select_index)
            decoder_inputs[x] = decoder_inputs[batch_index]
            decoder_inputs[x][index] = r_i[batch_index, select_index]
        print('Return', decoder_inputs[:,:index+1], nl)
        return decoder_inputs, nl
        #print("B", new_logits, logits, r_i)


    return r_i[index]

