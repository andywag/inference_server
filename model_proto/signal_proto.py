
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

def convert_type(typ:np.ndarray):
    if typ.dtype == np.int32:
        return "TYPE_INT32"
    elif typ.dtype == np.int64:
        return "TYPE_INT64"
    elif typ.dtype == np.uint32:
        return "TYPE_UINT32"
    elif typ.dtype == np.uint64:
        return "TYPE_UINT64"
    elif typ.dtype == np.float32:
        return "TYPE_FP32"

@dataclass
class SignalProto:
    name:str
    signal:np.ndarray
    random_data:np.ndarray = None
    

    def __post_init__(self):
        pass

    #def get_triton_type(self) -> str:
    #    return np_to_triton_dtype(self.signal.dtype)

    def get_shape(self, batch_size:Optional[int]=None)->List[int]:
        shape = list(self.signal.shape)
        if batch_size is not None:
            shape.insert(0,batch_size)
        return list(shape)

    def get_random_signal(self, batch_size:Optional[int]=None)->np.ndarray:
        if self.random_data is None:   
            self.random_data = np.random.rand(*self.get_shape(batch_size)).astype(self.signal.dtype)

        return self.random_data

    def create_signal(self, fptr, iotype="input"):
        fptr.write(f"{iotype} [\n")
        fptr.write("  {\n")
        fptr.write(f'    name:"{self.name}"\n')
        fptr.write(f"    data_type: {convert_type(self.signal)}\n")
        fptr.write(f"    dims: {list(self.signal.shape)}\n")
        fptr.write("  }\n")
        fptr.write(f"]\n\n")