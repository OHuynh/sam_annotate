from abc import abstractmethod


class Detector:
    def __init__(self, cap):
        self.cap = cap
        self._predictions = None

    @abstractmethod
    def interpolate(self, seq, label):
        pass
