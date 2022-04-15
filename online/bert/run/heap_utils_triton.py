
import heapq
from queue import Queue, Empty
import numpy as np
from packing_utils_triton import DataSpec
from functools import reduce
from collections import deque
import sys
import time
from deque_block_new import BlockingDeque

import logging
log = logging.getLogger(__name__)
from threading import Lock
import threading

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
        self.data['input_ids'] = self.indices
        self.data['position_ids'] = self.positions
        self.data['segment_ids'] = self.segments
        self.data['input_mask'] = self.mask_inputs



    def put(self, input_ids, first_segment, index, d):
        batch = index % self.w
        self.indices[batch, d.row, d.col:d.col + d.l] = input_ids[:d.l]
        self.positions[batch, d.row, d.col:d.col + d.l] = np.arange(d.l)
        c_segment_ids = np.ones(d.l, dtype=np.uint32)
        c_segment_ids[:first_segment] = np.zeros(first_segment, dtype=np.uint32)
        self.segments[batch, d.row, d.col:d.col + d.l] = c_segment_ids
        
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
        self.has_data = False

    def insert(self, id, length, sender_id, flush = False):
        self.has_data = True
        if length <= self.cols[0].width: # If the element fits in the stack push it into the heap
        #if self.cols[0].width == self.h:
            spec = DataSpec(id, self.cols[0].index, self.h - self.cols[0].width, length, flush = flush, sender = sender_id)
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
        #log.info(f"Getting Block{ len(self.run_block_queue)}")
        return self.run_block_queue.get_flush()

    def get_tensor(self, tensor_id):
        return self.input_buffer.get(tensor_id)

    
    def insert(self, input_ids, segment_ids, id, sender_id, flush = False):
        if flush:
            id = 0

        def insert_new_block(length):
            self.index = self.index + 1
            new_block = Block(self.b, self.h, self.index, last = flush)
            spec = new_block.insert(id, length, sender_id, flush = flush)
            self.input_buffer.put(input_ids, segment_ids, self.index, spec)
            self.run_block_queue.put(new_block)
        #    log.info(f"Inserted New Block { len(self.run_block_queue)} {sender_id}")

        inserted = False
        
        length = np.count_nonzero(input_ids)
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
        if queue_length > 16:
            start = queue_length - 15

        for x in range(start, queue_length):
            spec = self.run_block_queue[x].insert(id, length, sender_id = sender_id)
            if spec is not None and not flush:
                self.input_buffer.put(input_ids, segment_ids, self.index - queue_length + x + 1, spec)
                inserted = True
                return (True, False, self.index - queue_length + x + 1)

        if not inserted:
            insert_new_block(length)
        
        return (False, True, self.index)

    def insert_block(self, transfer):
        input_ids = transfer[0]
        segment_ids = transfer[1] 
        ids = transfer[2]
        sender_id = transfer[3]
        for x in range(len(input_ids)):
            self.insert(input_ids[x], segment_ids[x][0], ids[x], sender_id, False)

        

class PackingQueueHolder:
    def __init__(self, b, h, internal_buffer_width=64, input_queue_size = 8, input_queue_start = 4):
        self.b = b
        self.h = h
        self.index = 0
        self.final_index = 0
        self.input_queue_size = input_queue_size
        #self.run_block_queue = BlockingDeque(max_depth = input_queue_size, prime_depth = input_queue_start)
        #self.run_block_queue.put(Block(self.b, self.h, self.index))
        self.run_block_queue = Queue()
        self.final_queue = Queue(maxsize=1)

        self.input_buffer = CircularInputBuffer(internal_buffer_width, b, h)
        self.total_depth = 0
        self.total_count = 0
        self.dropped = False

        self.flush_count = 0  
        self.waiting = True
        self.working = Block(self.b, self.h, self.index)

        try :
            self.transfer_thread = threading.Thread(target=self.move)
            self.transfer_thread.start()
        except:
            print("Failed")
        print("Transfer Running")

    def move(self):
        while True:
            data = self.run_block_queue.get()
            self.final_queue.put(data)
           


    def get_legend(self):
        #time.sleep(.02)
        #print("Getting Legend", self.run_block_queue.qsize())
        #if self.final_queue.qsize() == 0:
            

        if self.final_queue.qsize() == 0:
            if len(self.working.specs) > 0:
                old_working = self.working
                self.working = Block(self.b, self.h, self.index)
                self.index += 1
                return old_working
            if self.flush_count > 0:
                self.flush_count -= 1
                block = Block(self.b, self.h, self.index, last = True)
                self.index += 1
                #print("Flushing Empty Packet")
                return block
           
        
        self.waiting = True
        data = self.final_queue.get()
        self.waiting = False

        self.flush_count = 2
        return data

    def get_tensor(self, tensor_id):
        return self.input_buffer.get(tensor_id)

    
    def insert(self, input_ids, segment_ids, id, sender_id, flush = False):
        #print("Inserting Data")
        length = np.count_nonzero(input_ids)
        queue_length = self.run_block_queue.qsize()
        self.total_depth += queue_length
        self.total_count += 1

        spec = self.working.insert(id, length, sender_id = sender_id)
        if spec is None:
            self.run_block_queue.put(self.working)
            self.working = Block(self.b, self.h, self.index)
            spec = self.working.insert(id, length, sender_id = sender_id)
            self.index += 1
        self.input_buffer.put(input_ids, segment_ids, self.index, spec)
        if self.waiting:
            self.run_block_queue.put(self.working)
            self.working = Block(self.b, self.h, self.index)
            self.index += 1
            
        #try:
        #self.run_block_queue.put(self.working)
        #self.working = Block(self.b, self.h, self.index)
        #self.index += 1
        #except queue.Full:
        #    pass
            
        

        return (True, False, self.index)

    def insert_block(self, transfer):
        input_ids = transfer[0]
        segment_ids = transfer[1] 
        ids = transfer[2]
        sender_id = transfer[3]
        for x in range(len(input_ids)):
            self.insert(input_ids[x], segment_ids[x][0], ids[x], sender_id, False)

        

    


  
