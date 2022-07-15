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
    


def create_data_internal_order(query_samples, idx, b, s, qsl):
    input_data = dict()
    input_data[INPUT_IDS] = np.zeros((b, s), dtype=np.uint32)
    input_data[POSITION_IDS] = np.zeros((b, s), dtype=np.uint32)
    input_data[SEGMENT_IDS] = np.zeros((b, s), dtype=np.uint32)
    input_data[INPUT_MASK] = np.ones((b, s), dtype=np.uint32)

    spec = []
    col_idx = 0
    row_idx = 0
    mask_idx = 0

    def insert(dl):
        input_data[INPUT_IDS][row_idx, col_idx:col_idx + dl] = eval_features.input_ids[:dl]
        input_data[POSITION_IDS][row_idx, col_idx:col_idx + dl] = np.arange(dl)
        input_data[SEGMENT_IDS][row_idx, col_idx:col_idx + dl] = eval_features.segment_ids[:dl]
        input_data[INPUT_MASK][row_idx, col_idx:col_idx + dl] = mask_idx * np.ones(dl, dtype=np.uint32)

    while row_idx < b and idx < len(query_samples):
        eval_features = qsl.get_features(query_samples[idx].index)
        data_len = int(np.count_nonzero(eval_features.input_ids))
        #if data_len >= 128:
        #    data_len = 128

        if col_idx + data_len <= s:
            insert(data_len)
            spec.append(DataSpec(query_samples[idx].id, row_idx, col_idx, data_len, idx))
            col_idx += data_len
            idx = idx + 1
            mask_idx = mask_idx + 1
            pass
        else:
            input_data[INPUT_MASK][row_idx, col_idx:] = mask_idx
            col_idx = 0
            row_idx = row_idx + 1
            mask_idx = 0
            pass
    return DataTransfer(idx, spec, input_data)

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

def create_input_data_pack(b,s):
    input_data = dict()
    input_data[INPUT_IDS] = np.zeros((b, s), dtype=np.uint32)
    input_data[POSITION_IDS] = np.zeros((b, s), dtype=np.uint32)
    input_data[SEGMENT_IDS] = np.zeros((b, s), dtype=np.uint32)
    input_data[INPUT_MASK] = np.zeros((b, s), dtype=np.uint32)
    return input_data

def find_row_greedy(b, s, data_len, col_idx):
    for x in range(b):
        if col_idx[x] + data_len <= s:
            return True, x
    return False, x
    

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

def compress_data(input_ids, segment_ids, data_len, ratio):
    sel = np.random.uniform(low=0.0, high=1.0, size=(data_len))
    ind = np.where(sel <= ratio)
    uind = ind[0]

    return input_ids[uind], uind, segment_ids[uind]

def compress_length(input_ids, segment_ids, data_len, ratio):
    uind = np.arange(int(ratio*len(input_ids)))

    return input_ids[uind], uind, segment_ids[uind]

def compress_tokens(input_ids, segment_ids, data_len, tokens):
    uind = np.arange(int(1.0*len(input_ids)))
    for x in range(len(input_ids)):
        if False:
            print("Deleting Token", x)
            np.delete(uind,x)
            import sys
            sys.exit(0)
    return input_ids[uind], uind, segment_ids[uind]


def create_array(eval_features_list, specs, b, s):
    input_data = create_input_data_pack(b,s)
    idx = 0
    mask_idx = [1 for x in range(b)]

    for spec in specs:
        eval_features = eval_features_list[idx]
        insert(input_data, spec.l, spec.row, spec.col, mask_idx[spec.row], eval_features.input_ids, np.arange(spec.l), eval_features.segment_ids)
        idx += 1
        mask_idx[spec.row] += 1

    return input_data






def no_pack_data_triton(input_ids, segment_ids, query_ids,  b, s, flush = False):
    
    input_data = create_input_data_pack(b,s)
    specs = []
    for x in range(len(input_ids)):
        if x == 0:
            print("Adding Data", query_ids[x])
        data_len = int(np.count_nonzero(input_ids[x]))
        first_segment = int(segment_ids[x])

        input_data[INPUT_IDS][x] = input_ids[x,:]
        input_data[SEGMENT_IDS][x,first_segment:] = np.ones((s-first_segment))
        input_data[POSITION_IDS][x] = np.arange(s)
        input_data[INPUT_MASK][x,data_len:] = np.ones((s-data_len)) 
        
        specs.append(DataSpec(query_ids[x], x, 0, data_len, x))


    return DataTransfer(0, specs, input_data, flush)


class Packer:
    def __init__(self, b, s, run_queue):
        self.b = b
        self.s = s
        self.run_queue = run_queue
        self.create_packet()

    def create_packet(self):
        print("Creating Packet")
        self.input_data = create_input_data(self.b, self.s)
        self.spec = []
        self.row_idx = 0
        self.col_idx = [0 for x in range(self.b)]
        self.mask_idx = [0 for x in range(self.b)]
        self.count = 0
    
    def pack_data_triton_async(self, input_id, segment_id, query_id):

        data_len = int(np.count_nonzero(input_id))
        c_segment_ids = np.ones(data_len, dtype=np.uint32)
        c_segment_ids[:int(segment_id)] = np.zeros(int(segment_id), dtype=np.uint32)

        print("Here", data_len, input_id)

        found, x = find_row_full(self.b,self.s,data_len,self.col_idx)
        #print("Packing", found, x)
        if not found:
            print("Created Transfer")
            data = DataTransfer(self.count, self.spec, self.input_data)
            self.run_queue.put(data)
            self.create_packet()
            found, x = find_row_full(self.b,self.s,data_len,self.col_idx)

        insert(self.input_data, data_len, x, self.col_idx[x], self.mask_idx[x], 
                input_id, np.arange(data_len), c_segment_ids)
        self.spec.append(DataSpec(query_id, x, self.col_idx[x], data_len))
        self.col_idx[x] += data_len
        self.mask_idx[x] += 1
        print("Done with Transfer")

            
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


def pack_data_triton(input_ids, segment_ids, query_ids,  idx, b, s):
    
    input_data = create_input_data(b,s)

    spec = []
    row_idx = 0
    col_idx = [0 for x in range(b)]
    mask_idx = [0 for x in range(b)]

    tic = time.time()
    while idx < len(input_ids):
        #eval_features = qsl.get_features(query_samples[idx].index)
        data_len = int(np.count_nonzero(input_ids[idx]))
        c_input_ids = np.asarray(input_ids[idx,:data_len], dtype=np.uint32)
        positions = np.arange(data_len, dtype=np.uint32)
        
        first_segment = int(segment_ids[idx])
        c_segment_ids = np.ones(data_len, dtype=np.uint32)
        c_segment_ids[:first_segment] = np.zeros(first_segment, dtype=np.uint32)

        data_len = int(np.count_nonzero(c_input_ids))

        found, min_idx = find_row_full(b,s,data_len,col_idx)

        if found or col_idx[x] == 0:
            x = min_idx            

            insert(input_data, data_len, x, col_idx[x], mask_idx[x], c_input_ids, np.arange(data_len), c_segment_ids)
            spec.append(DataSpec(query_ids[idx], x, col_idx[x], data_len))
            
            col_idx[x] += data_len
            idx = idx + 1
            mask_idx[x] = mask_idx[x] + 1
                
        if not found:
            break
    for x in range(b):
        input_data[INPUT_MASK][x,col_idx[x]:] = mask_idx[x]

    total_fill = 0
    for x in range(b):
        total_fill += col_idx[x]
       

    return DataTransfer(idx, spec, input_data), total_fill/384/b


def pack_data(query_samples,  idx, b, s, qsl, greedy=False, pack_group=None):
    
    input_data = create_input_data(b,s)

    spec = []
    row_idx = 0
    col_idx = [0 for x in range(b)]
    mask_idx = [0 for x in range(b)]

    tic = time.time()
    while idx < len(query_samples):
        eval_features = qsl.get_features(query_samples[idx].index)
        data_len = int(np.count_nonzero(eval_features.input_ids))
        input_ids = np.asarray(eval_features.input_ids[:data_len], dtype=np.uint32)
        positions = np.arange(data_len, dtype=np.uint32)
        segment_ids = np.asarray(eval_features.segment_ids[:data_len], dtype=np.uint32)

        odata_len = data_len
           

        data_len = int(np.count_nonzero(input_ids))


        #limit = int(len(spec))
        if greedy:
            found, min_idx = find_row_greedy(b,s,data_len,col_idx)
        else:
            found, min_idx = find_row_full(b,s,data_len,col_idx)


        if found or col_idx[x] == 0:
            x = min_idx            

            insert(input_data, data_len, x, col_idx[x], mask_idx[x], input_ids, np.arange(data_len), segment_ids)
            spec.append(DataSpec(query_samples[idx].id, x, col_idx[x], data_len))
            
            col_idx[x] += data_len
            idx = idx + 1
            mask_idx[x] = mask_idx[x] + 1
                
        if not found:
            break
    for x in range(b):
        input_data[INPUT_MASK][x,col_idx[x]:] = mask_idx[x]

    total_fill = 0
    for x in range(b):
        total_fill += col_idx[x]
    #print("Fill", idx, total_fill/384/b, time.time() - tic)
       

    return DataTransfer(idx, spec, input_data), total_fill/384/b


def run_internal_ipu(runner, input_data):
    tic = time.time()
    print("Starting to Run")
    results = runner.run(input_data.data)
    print("Batch", time.time() - tic)
    #import sys
    #sys.exit(0)
    return DataTransfer(input_data.count, input_data.ids, input_data.total, results)

def run_ipu(runner, input_queue, output_queue):
    while True:
        input_data = input_queue.get()
        if input_data is None:
            output_queue.put(None)
            print("Ending Run Thread")
            break
        result = run_internal_ipu(runner, input_data)
        output_queue.put(result)

