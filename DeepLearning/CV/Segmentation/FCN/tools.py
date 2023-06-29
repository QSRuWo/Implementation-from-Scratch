import time

class Accumulator:
    '''
    For accumulating sums over n variables
    '''
    def __init__(self, n):
        self.data = [0.] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Timer:
    '''
    Record Multiple Running Times
    '''
    def __init__(self):
        self.time = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.time.append(time.time() - self.tik)
        return self.time[-1]

    def avg(self):
        return sum(self.time) / len(self.time)

    def sum(self):
        return sum(self.time)

    def cumsum(self):
        return np.array(self.time).cumsum().tolist()