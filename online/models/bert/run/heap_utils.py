
import heapq
from queue import Queue
import numpy as np
from packing_utils import DataSpec
from functools import reduce
from collections import deque
#from deque_block import BlockingDeque
import sys
import time
from deque_block_new import BlockingDeque

class CircularOutputBuffer:
    def __init__(self, w, vector, run_queue, output_queue, replication = 1):
        self.vector = vector['Squad/Gemm:0']
        self.w = w
        #self.index = 0
        self.run_queue = run_queue
        self.output_queue = output_queue
        self.tensor_id = None
        self.input_index = 0
        self.output_index = 0

    def get_vector(self, tensor_id, index):
        vector = self.vector[tensor_id][index % self.w]
        return vector

    def get(self, tensor_id):
        #print(tensor_id)
        #self.tensor_id = tensor_id
        vector = self.vector[ self.input_index % self.w]
        self.input_index += 1
        return vector

    def get_complete(self):
        try :
            block = self.run_queue.get(False)
            vector = self.vector[ self.output_index % self.w]
            self.output_index += 1
            self.output_queue.put( (vector, block.specs) )
        except Exception as e:
            pass

    
class CircularInputBuffer:
    def __init__(self, w, b, h):
        self.first = None
        self.index = -1
        self.w = w
        self.b = b
        self.h = h
        self.indices = np.zeros((w, b, h), dtype = np.uint32)
        self.positions = np.zeros((w, b, h), dtype = np.uint32)
        self.segments = np.zeros((w, b, h), dtype = np.uint32)
        self.mask_inputs = np.zeros((w, b, h), dtype = np.uint32)

        self.data = dict()
        self.data['indices'] = self.indices
        self.data['positions'] = self.positions
        self.data['segments'] = self.segments
        self.data['input_mask'] = self.mask_inputs



    def put(self, data, index, d):
        batch = index % self.w
        self.indices[batch, d.row, d.col:d.col + d.l] = data.input_ids[:d.l]
        self.positions[batch, d.row, d.col:d.col + d.l] = np.arange(d.l)
        self.segments[batch, d.row, d.col:d.col + d.l] = data.segment_ids[:d.l]
        
        if d.col == 0:
            self.mask_inputs[batch, d.row, :] = np.zeros(self.h, dtype=np.uint32)
            mask = 1
        else:
            mask = self.mask_inputs[batch, d.row, d.col-1]+1

        self.mask_inputs[batch, d.row, d.col:d.col + d.l] = mask * np.ones(d.l, dtype=np.uint32)

    def get(self, tensor_id):

        if self.first is None:
            self.first = tensor_id

        if self.first == tensor_id:
            self.index = (self.index + 1) % self.w
          
        return self.data[tensor_id][self.index ,:,:]

       


# TODO : Remove class basic wrapper for single int
class Element:
    def __init__(self, index, width):
        self.index = index
        self.width = width

    def __repr__(self):
        return f"({self.index},{self.width})"

    def __str__(self):
        return f"({self.index},{self.width})"

    def update(self, nw):
        self.width = self.width - nw
        return self #Element(self.index, self.width)

    def __lt__(self, other):
        return self.width > other.width

# 
# Class which describes a block of data and contains an index of elements contained inside
# The internal data structure contains a heap which describes if the next input sample can fit inside

class Block:
    def __init__(self, b, h, index, last = False):
        self.h = h
        self.cols = [Element(x, h) for x in range(b)]
        self.specs = []
        self.last = last

    def insert(self, id, length, flush = False):
        if length <= self.cols[0].width: # If the element fits in the stack push it into the heap
            spec = DataSpec(id, self.cols[0].index, self.h - self.cols[0].width, length, flush = flush)
            self.specs.append(spec)
            heapq.heapreplace(self.cols, self.cols[0].update(length))
            return spec
        else:
            return None


class PackingQueueHolderNew:
    def __init__(self, b, h, internal_buffer_width=64, input_queue_size = 8, input_queue_start = 4):
        self.b = b
        self.h = h
        self.index = 0
        self.input_queue_size = input_queue_size
        self.run_block_queue = BlockingDeque(max_depth = input_queue_size, prime_depth = input_queue_start)
        self.run_block_queue.put(Block(self.b, self.h, self.index))
        self.input_buffer = CircularInputBuffer(internal_buffer_width, b, h)
        self.total_depth = 0
        self.total_count = 0
        self.dropped = False

    def get_legend(self):
        return self.run_block_queue.get_flush()

    def get_tensor(self, tensor_id):
        return self.input_buffer.get(tensor_id)

    def insert(self, id, data, flush = False):
        if flush:
            id = 0

        def insert_new_block(length):
            self.index = self.index + 1
            new_block = Block(self.b, self.h, self.index, last = flush)
            spec = new_block.insert(id, length, flush = flush)
            self.input_buffer.put(data, self.index, spec)
            self.run_block_queue.put(new_block)

        inserted = False
        
        length = np.count_nonzero(data.input_ids)
        queue_length = len(self.run_block_queue)
        self.total_depth += queue_length
        self.total_count += 1

        if self.dropped or queue_length > int(.8*self.input_queue_size):
            print("Fifo Overflow")
            self.dropped = True
            #sys.exit(1)
        
        if flush:
            insert_new_block(length)
            

        start = 0
        if queue_length > 2:
            start = queue_length - 3

        for x in range(start, queue_length):
            spec = self.run_block_queue[x].insert(id, length)
            if spec is not None and not flush:
                self.input_buffer.put(data, self.index - queue_length + x + 1, spec)
                inserted = True
                return (True, False, self.index - queue_length + x + 1)

        if not inserted:
            insert_new_block(length)
        
        return (False, True, self.index)


        

class PackingQueueHolder:
    def __init__(self, b, h, input_buffer, run_block_queue):
        self.b = b
        self.h = h
        self.index = 0
        self.run_block_queue = run_block_queue
        self.run_block_queue.put(Block(self.b, self.h, self.index))
        self.input_buffer = input_buffer
        self.dropped = False



    def insert(self, id, data, flush = False):

        def insert_new_block(length):
            self.index = self.index + 1
            new_block = Block(self.b, self.h, self.index, last = flush)
            spec = new_block.insert(id, length, flush = flush)
            self.input_buffer.put(data, self.index, spec)
            self.run_block_queue.put(new_block)

        inserted = False
        
        length = np.count_nonzero(data.input_ids)
        queue_length = len(self.run_block_queue)
        
        if self.dropped or queue_length > 16:
            print("Fifo Overflow")
            self.dropped = True

        if flush:
            insert_new_block(length)
            

        for x in range(queue_length):
            #print("Pushing value", self.index, x, len(self.run_block_queue))
            spec = self.run_block_queue[x].insert(id, length)
            if spec is not None and not flush:
                self.input_buffer.put(data, self.index - queue_length + x + 1, spec)
                inserted = True
                break

        if not inserted:
            insert_new_block(length)

  
