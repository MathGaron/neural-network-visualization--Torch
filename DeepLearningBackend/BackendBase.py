from abc import abstractmethod


class BackendBase:
    def __init__(self, processing_backend):
        self.processing_backend = processing_backend

    @abstractmethod
    def predict(self, input):
        pass

    @abstractmethod
    def get_convolution_activation(self):
        pass

