require 'torch'
require 'image'
require 'gnuplot'
require 'nn'
require 'loadcaffe'

local Flashlight = torch.class('Flashlight')

function Flashlight:__init(backend)
    self.backend = backend
    if self.backend == "cuda" then
        require 'cunn'
    end
    self.truncated_nets = {}

end

function Flashlight:build_model()

    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(3, 64, 
        5, 5,
        1, 1))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    net:add(nn.SpatialConvolution(64, 128, 
        3, 3,
        2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.Dropout(0.2))
    -- Inception Module
    reductions = {
        64,
        64,
        32,
        128
    }
    expansions = {
        256,
        64
    }
    net:add(self:inception_module(128, 512, reductions, expansions))
    net:add(nn.SpatialConvolution(512, 768, 3, 3, 1, 1))
    net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
    -- Inception Module
    reductions = {
        64,
        256,
        256,
        128
    }
    expansions = {
        320,
        512
    }
    net:add(self:inception_module(768, 1024, reductions, expansions))
    net:add(nn.SpatialAveragePooling(5, 5, 1, 1))
    net:add(nn.View(1024))
    net:add(nn.Linear(1024, 512))
    net:add(nn.Dropout(0.4))
    net:add(nn.Linear(512, 256))
    net:add(nn.Dropout(0.4))
    net:add(nn.Linear(256, 10))
    --print(net)
    self.net = self:set_backend(net)
end

function Flashlight:set_backend(module)
    if self.backend == "cuda" then
        return module:cuda()
    else
        return module:float()
    end
end

function Flashlight:load(path)
    local net = torch.load(path)
    self:remove_batch_norm(net)
    print(net)
    self.net = self:set_backend(net)
end

function Flashlight:load_caffe_model(model, weights)
    self.net = loadcaffe.load(model, weights)
    self:remove_batch_norm(self.net)
    print(self.net)
    self.net = self:set_backend(self.net)
end

function Flashlight:remove_batch_norm(net)
    for i, module in ipairs(net.modules) do
        if torch.type(module) == 'nn.SpatialBatchNormalization' then
            net:remove(i)
        end
    end
end

function Flashlight:predict(image)
    image = self:set_backend(image)
    self.net:evaluate()
    self.last_input = image
    local output = self.net:forward(image)
    return output:float()
end

function Flashlight:backward(forward_output)
    forward_output = self:set_backend(forward_output)
    return self.net:backward(self.last_input, forward_output):float()
end

function Flashlight:truncate_network(net, index)
    local delete = false
    for i, curModule in ipairs(self.net.modules) do
        if delete then
            net:remove()
        end
        if curModule['LAYER_INDEX'] == index then
            delete = true
        end
    end
    return net
end

function Flashlight:backward_layer(activation_output, index)
    if self.truncated_net == nil or self.truncated_index ~= index then
        self.truncated_net = self.net:clone('weight','bias');
        self.truncated_index = index
        self:truncate_network(self.truncated_net, index)
    end
    activation_output = self:set_backend(activation_output)
    local grad = self.truncated_net:backward(self.last_input, activation_output)
    return grad:float()
end

function Flashlight:get_activation()
    self.filterResponses = {}
    count = 0
    for i, curModule in ipairs(self.net.modules) do
        if torch.type(curModule) == "nn.ReLU" then
            curModule['LAYER_INDEX'] = count
            count = count + 1
            table.insert(self.filterResponses, curModule.output:float())
        end
    end
    return self.filterResponses
end



-- Close all gnuplot windows
function Flashlight:clear_gnu_plots()
    os.execute('pkill gnuplot')
end
