import numpy as np
import time

INPUT_IDS = "input_ids"
POSITION_IDS = "position_ids"
SEGMENT_IDS = "segment_ids"
INPUT_MASK = "input_mask"

class DataSpec:
    def __init__(self, id, row, col, length, positions = None, flush = False, sender = 0):
        self.id = id
        self.row = row
        self.col = col
        self.l = length
        self.positions = positions
        self.flush = flush
        self.sender = sender

    def __str__(self):
        return f"{self.id},{self.row},{self.col},{self.l}"
        
    def __repr__(self):
        return f"{self.id},{self.row},{self.col},{self.l}"

    def debug(self):
        return f"{self.id},{self.row},{self.col},{self.l}"

    def shift(self, row_offset):
        return DataSpec(self.id, self.row - row_offset, self.col, self.l)

class DataTransfer:
    def __init__(self, count, spec, data, last = False):
        self.count = count
        self.specs = spec
        self.data = data
        self.last = last

    def combine(self, data):
        new_specs = self.specs + data.specs
        new_data = dict()
        new_data[INPUT_IDS] = np.concatenate((self.data[INPUT_IDS], data.data[INPUT_IDS]),axis=0)
        new_data[POSITION_IDS] = np.concatenate((self.data[POSITION_IDS], data.data[POSITION_IDS]),axis=0)
        new_data[SEGMENT_IDS] = np.concatenate((self.data[SEGMENT_IDS], data.data[SEGMENT_IDS]),axis=0)
        new_data[INPUT_MASK] = np.concatenate((self.data[INPUT_MASK], data.data[INPUT_MASK]),axis=0)

        return DataTransfer(self.count, new_specs, new_data)

    def update(self, data):
        return DataTransfer(self.count, self.specs,  data, self.last)
    
    def flush(self):
        return DataTransfer(self.count, self.specs,  self.data, True)

    def split(self, groups, total_rows):
        size_of_group = int(total_rows/groups)
        new_specs = [[] for x in range(groups)]

        for spec in self.specs:
            gr = int(spec.row/size_of_group)
            new_specs[gr].append(spec.shift(gr*size_of_group))

        transfers = []
        for x in range(groups):
            new_data = dict()
            new_data[INPUT_IDS] =self.data[INPUT_IDS][x*size_of_group:(x+1)*size_of_group]
            new_data[POSITION_IDS] =self.data[POSITION_IDS][x*size_of_group:(x+1)*size_of_group]
            new_data[SEGMENT_IDS] =self.data[SEGMENT_IDS][x*size_of_group:(x+1)*size_of_group]
            new_data[INPUT_MASK] =self.data[INPUT_MASK][x*size_of_group:(x+1)*size_of_group]
            transfers.append(DataTransfer(self.count,new_specs[x],new_data))
        return transfers

    def debug(self):
        return f"{self.count} {len(self.specs)}"
    



def insert(input_data, dl, row, col, mask, input_ids, positions, segment_ids):
    input_data[INPUT_IDS][row, col:col + dl] = input_ids[:dl]
    input_data[POSITION_IDS][row, col:col + dl] = positions
    input_data[SEGMENT_IDS][row, col:col + dl] = segment_ids[:dl]
    input_data[INPUT_MASK][row, col:col + dl] = mask * np.ones(dl, dtype=np.uint32)




def create_input_data(b,s):
    input_data = dict()
    input_data[INPUT_IDS] = np.zeros((b, s), dtype=np.uint32)
    input_data[POSITION_IDS] = np.zeros((b, s), dtype=np.uint32)
    input_data[SEGMENT_IDS] = np.zeros((b, s), dtype=np.uint32)
    input_data[INPUT_MASK] = np.ones((b, s), dtype=np.uint32)
    return input_data


    

def find_row_full(b, s, data_len, col_idx):
    found = False
    min_idx = None
    min_err = 1000
    for x in range(b):
        err = s - data_len - col_idx[x]
        if err >= 0:
            if min_idx is None or (err > 0 and err < min_err):
                min_idx = x
                min_err = err
                found = True
            if err < 32:
                break
    return found, min_idx
     
def empty_data_transfer(b,s):
    input_data = create_input_data(b,s)
    return DataTransfer(0, [], input_data)

def pack_data_triton_queue(input_queue, b, s, last = None):
    
    input_data = create_input_data(b,s)

    spec = []
    row_idx = 0
    col_idx = [0 for x in range(b)]
    mask_idx = [0 for x in range(b)]

    tic = time.time()
    while True:
        if input_queue.empty() and len(spec) > 0:
            print("Breaking from Transfer Loop", input_queue.qsize(), len(spec))
            return DataTransfer(0, spec, input_data), None
        
        if last is not None:
            c_input_ids, first_segment, data_id, sender = last
            last = None
        else:
            c_input_ids, first_segment, data_id, sender = input_queue.get()
        
        first_segment = int(first_segment[0])
        #eval_features = qsl.get_features(query_samples[idx].index)
        data_len = int(np.count_nonzero(c_input_ids))
        c_input_ids = np.asarray(c_input_ids[:data_len], dtype=np.uint32)
        positions = np.arange(data_len, dtype=np.uint32)        
        c_segment_ids = np.ones(data_len, dtype=np.uint32)
        c_segment_ids[:first_segment] = np.zeros(first_segment, dtype=np.uint32)

        found, min_idx = find_row_full(b,s,data_len,col_idx)

        if found or col_idx[x] == 0:
            x = min_idx            

            insert(input_data, data_len, x, col_idx[x], mask_idx[x], c_input_ids, np.arange(data_len), c_segment_ids)
            spec.append(DataSpec(data_id, x, col_idx[x], data_len, sender = sender))
            
            col_idx[x] += data_len
            mask_idx[x] = mask_idx[x] + 1
        #print("HHHH", input_queue.qsize())
        if not found:

            break

    #print("Packet Done")
    for x in range(b):
        input_data[INPUT_MASK][x,col_idx[x]:] = mask_idx[x]

    return DataTransfer(0, spec, input_data), (c_input_ids, [first_segment], data_id, sender)

