import PyTorchHelpers


class TorchBackend:
    def __init__(self,backend="cuda"):
        # load lua files
        Flashlight = PyTorchHelpers.load_lua_class("torch-nn-viz-example.lua", 'Flashlight')
        self.model = Flashlight(backend)
        self.model.clear_gnu_plots()

    def load_cafe_model(self, prototxt, caffemodel):
        self.model.load_caffe_model(prototxt, caffemodel)

    def predict(self, input):
        return self.model.predict(input).asNumpyTensor()

    def get_convolution_activation(self):
        filters = self.model.get_convolution_activation()
        filters = TorchBackend.torch2numpy_(filters)
        filters = TorchBackend.dict2list_(filters)
        return filters

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
