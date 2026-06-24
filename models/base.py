from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()

    @abstractmethod
    def _build_model(self):
        pass
        
    def forward(self, x):
        return self.model(x)
    