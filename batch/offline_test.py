

import pickle
import numpy as np 

with open('temp.pik','rb') as fp:
    result = pickle.load(fp)

print(len(result))
print(result[0][0])

