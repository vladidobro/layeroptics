from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def propagation(self, k):
        return NotImplemented

