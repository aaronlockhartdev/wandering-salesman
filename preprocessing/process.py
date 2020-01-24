import numpy as np

class Processor():
    def __init__(self, batchSize):
        self.batchSize = batchSize
        
    def __call__(self):
        boards = np.random.rand(100000, self.batchSize, 10, 10, 1)
        starts = np.random.rand(100000, self.batchSize, 2)
        starts = starts * 10
        starts = starts.astype(np.int)
        ends = np.random.rand(100000, self.batchSize, 2)
        ends = ends * 10
        ends = ends.astype(np.int)

        return boards, starts, ends