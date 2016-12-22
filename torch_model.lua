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
    -- backend transfer
    forward_output = self:set_backend(forward_output)
    return self.net:backward(self.last_input, forward_output):float() -- L2 distance for grad
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

function Flashlight:backward_test(forward_inputs)
    forward_inputs = self:set_backend(forward_inputs)
    local modules_n = #self.net.modules
    temp = forward_inputs:clone()
    for i=modules_n,1,-1 do
        grad = self.net.modules[i]:backward(self.last_input, temp)
        tmp = grad:clone()
        print(i)
        print(torch.type(self.net.modules[i]))
        --temp = grad:clone()
    end
    return 0
end

-- Retrieve the filter responses caused by passing the image through the model
-- Each table of filter responses from a layer has a field 'ADDED_NAME' 
-- added to it which contains the name of the layer type. This is to make
-- reviewing filter responses and mapping them back to layers easier...
-- Return the filter responses in a table 
function Flashlight:get_convolution_activation()
    self.filterResponses = {}
    count = 0
    for i, curModule in ipairs(self.net.modules) do
        if torch.type(curModule) == "nn.SpatialConvolution" then
            curModule['LAYER_INDEX'] = count
            count = count + 1
            table.insert(self.filterResponses, curModule.output:float())
        end
    end
    return self.filterResponses
end

function Flashlight:get_convolution_filters()
    self.filterWeights = {}
    for i, curModule in ipairs(self.net.modules) do
        curModule['ADDED_NAME'] = torch.type(curModule)
        if curModule.ADDED_NAME == "nn.SpatialConvolution" then
            local filter = curModule.weight.new()
            filter:resize(curModule.weight:nElement())
            filter:copy(curModule.weight)
            table.insert(self.filterWeights, filter:float())
            --print(curModule.weight:size()) : first layer : 96*3*7*7, next : 256*96*5*5
        end
    end
    --return self.filterWeights
end

-- Close all gnuplot windows
function Flashlight:clear_gnu_plots()
    os.execute('pkill gnuplot')
end
