import threading
import queue
import traceback



class ThreadHelper:
    def __init__(self, fn, name):
        self.fn = fn
        self.name = name
        self.queue = queue.Queue()
        threading.Thread(target = self.run, args=[], name = name).start()

    def put(self, data, last = False):
        self.queue.put((data, last))

    def get(self):
        return self.queue.get()

    def run(self):
        while True:
            try :
                data = self.get()
                self.fn(data[0])
                if data[1]:             
                    break
                
            except Exception as e:
                print(self.name, e)
                traceback.print_exc()

        print("Finished with Thread",self.name)