import pickle
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

with open('tf_data.pkl','rb') as tf:
    data_tf = pickle.load(tf)

with open('torch_data.pkl','rb') as tf:
    data_torch = pickle.load(tf)

for x in data_tf.keys():
    err = data_tf[x] - data_torch[x]
    print(x,np.var(err))

#err = data_tf['Layer1/Attention/QKV'] - data_torch['Layer1/Attention/QKV']
#print(err.shape)
#print(np.var(err[:,0:]))