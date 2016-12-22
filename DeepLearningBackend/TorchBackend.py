import PyTorchHelpers
from DeepLearningBackend.BackendBase import BackendBase


class TorchBackend(BackendBase):
    def __init__(self, processing_backend="cuda"):
        super().__init__(processing_backend)
        Flashlight = PyTorchHelpers.load_lua_class("torch_model.lua", 'Flashlight')
        self.model = Flashlight(self.processing_backend)
        self.model.clear_gnu_plots()

    def load(self, path):
        self.model.load(path)

    def load_cafe_model(self, prototxt, caffemodel):
        self.model.load_caffe_model(prototxt, caffemodel)

    def forward(self, input):
        return self.model.predict(input).asNumpyTensor()

    def backward(self, grad):
        return self.model.backward(grad).asNumpyTensor()

    def backward_layer(self, grad, index):
        filters = self.get_convolution_activation()
        if grad.shape != filters[index].shape:
            raise IndexError("Gradient shape must be {}".format(filters[index].shape))
        grads = self.model.backward_layer(grad, index).asNumpyTensor()
        return grads

    def get_activation(self):
        filters = self.model.get_activation()
        filters = TorchBackend.torch2numpy_(filters)
        filters = TorchBackend.dict2list_(filters)
        convos = []
        linear = []
        for filter in filters:
            if len(filter.shape) > 2:
                convos.append(filter)
            else:
                linear.append(filter)
        return convos, linear

    def get_convolution_filters(self):
        weights = self.model.get_convolution_filters()
        print(weights)

    @staticmethod
    def torch2numpy_(data):
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = value.asNumpyTensor()
        return data

    @staticmethod
    def dict2list_(dict):
        index = [k for k in dict.keys()]
        index.sort()
        return [dict[i] for i in index]
