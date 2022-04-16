

from collections import deque
import queue

class BlockingDeque:
    def __init__(self, max_depth= 8, prime_depth = 4):
        self.max_depth = max_depth
        self.prime_depth = prime_depth
        self.queue = deque([],max_depth)
        self.primed = False
        if self.prime_depth == 0:
            self.primed = True

        self.output_queue = queue.Queue(maxsize = 1)

    def put(self, input):
        # Initial Condition to Allow the system to be emptied
        if not self.primed and len(self.queue) == self.prime_depth:
            self.primed = True

        # Load up the Output Queue if Empty
        if self.primed and self.output_queue.qsize() == 0 and len(self.queue) > 0:
            self.output_queue.put(self.queue.popleft())
            
        # Finally Append the Input to the Queue
        self.queue.append(input)

    # Blocks if the fifo is not primed other wise flushes out nones to allow the pipeline to clear
    def get_flush(self):
        #data = self.output_queue.get()
        #return data

        if self.output_queue.empty() and self.primed:
            if len(self.queue) > 0:
                data = self.queue.popleft()
                return data
            else:
                return None
        else :
            data = self.output_queue.get()
            return data
       


    
    def __len__(self):
        return len(self.queue)

    def __getitem__(self, key):
        return self.queue[key]


