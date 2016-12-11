from abc import abstractmethod


class BackendBase:
    def __init__(self, processing_backend):
        self.processing_backend = processing_backend

    @abstractmethod
    def forward(self, input):
        """
        Compute the model's forward step. store the activation function of each layers
        :param input: numpy array of the input
        :return: numpy array of the output
        """
        pass

    @abstractmethod
    def backward(self, loss):
        """

        :param loss:
        :return:
        """
        pass

    @abstractmethod
    def get_convolution_activation(self):
        """
        This function return the models activations after a forward pass of the convolution layers
        Returns a list of numpy array : [layer1, layer2, layer3 ... layern]
        numpy array have shape layern.shape = (1, nfilters, outputsize_w, outputsize_h)
        :return:
        """
        pass

