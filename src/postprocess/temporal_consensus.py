from collections import deque, Counter

class TemporalConsensus:
    def __init__(self, window=7):
        self.window = window
        self.history = {}

    def smooth(self, key, label):
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window)

        self.history[key].append(label)
        return Counter(self.history[key]).most_common(1)[0][0]
