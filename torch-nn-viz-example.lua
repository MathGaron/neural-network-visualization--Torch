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

-- Very Simple test for gnuplot
function Flashlight:gnuplot_test()
    a = image.lena();
    gnuplot.figure(1);
    gnuplot.imagesc(a[1])
end

-------------------------------------------------------------------------------
-----------------------TEST NETWORK DEFINITION---------------------------------
-------------------------------------------------------------------------------
-- Constructs and returns an inceptionModule from the paper 
-- "Going Deeper with Convolutional Networks", with input/output channels defined
-- with the parameters as follows:
-- inputChannels: the number of input channels
-- outputChannels: the expected number of outputChannels 
--                  (this parameter is only used to check the other parameters)
-- reductions: a 4-element array which specifies the number of channels output
--                  from each 1x1 convolutional network 
--                  (which should be smaller than the inputChannels usually...)
-- expansions: a 2-element array which specifies the number of channels output
--                  from the 3x3 convolutional layer and 
--                  the 5x5 convolutional layer
-- ReLU activations are applied after each convolutional layer
-- This module might be extended to allow for arbitrary width
function Flashlight:inception_module(inputChannels, outputChannels, reductions, expansions)

    computedOutputChannels = reductions[1] + expansions[1] + expansions[2] + reductions[4]
    if not (outputChannels == computedOutputChannels) then
        print("\n\nOUTPUT CHANNELS DO NOT MATCH COMPUTED OUTPUT CHANNELS")
        print('outputChannels: ', outputChannels)
        print('computedOutputChannels: ', computedOutputChannels)
        print("\n\n")
        return nil
    end

    -- Remember, if there is no stacked first dimension (which here is just a
    -- single entry in the first dimension) then this should be 1.
    -- But since I reshape and add the empty first dimension, 
    -- I can keep this as 2.
    local inception = nn.DepthConcat(2)

    local column1 = nn.Sequential()
    column1:add(nn.SpatialConvolution(inputChannels, reductions[1],
        1, 1,  -- Convolution kernel
        1, 1)) -- Stride
    column1:add(nn.ReLU(true))
    inception:add(column1)
    
    local column2 = nn.Sequential()
    column2:add(nn.SpatialConvolution(inputChannels, reductions[2],
        1, 1, 
        1, 1))
    column2:add(nn.ReLU(true))
    column2:add(nn.SpatialConvolution(reductions[2], expansions[1],
        3, 3,  -- Convolution kernel
        1, 1)) -- Stride
    column2:add(nn.ReLU(true))
    inception:add(column2)

    local column3 = nn.Sequential()
    column3:add(nn.SpatialConvolution(inputChannels, reductions[3],
        1, 1, 
        1, 1))
    column3:add(nn.ReLU(true))
    column3:add(nn.SpatialConvolution(reductions[3], expansions[2],
        5, 5,  -- Convolution kernel
        1, 1)) -- Stride
    column3:add(nn.ReLU(true))
    inception:add(column3)

    local column4 = nn.Sequential()
    column4:add(nn.SpatialMaxPooling(3, 3, 1, 1))
    column4:add(nn.SpatialConvolution(inputChannels, reductions[4],
        1, 1,  -- Convolution kernel
        1, 1)) -- Stride
    column4:add(nn.ReLU(true))
    inception:add(column4)

    return inception
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
    self.net = net
    if self.backend == "cpu" then
        self.net:float()
    else
        self.net:cuda()
    end

end

-------------------------------------------------------------------------------
------------------------VISUALIZATION FUNCTIONS--------------------------------
-------------------------------------------------------------------------------
-- Load a pre-trained model, remove all SpatialBatchNormalization layers
-- SpatialBatchNormalization requires batches of images, visualization 
-- explicitly uses only a single image
-- There may be other layer types that have this problem that I am unaware of
function Flashlight:load_model()
    local net = torch.load('model-nets/model--float.net')
    self:remove_batch_norm(net)
    print(net)
    self.net = net
    if self.backend == "cpu" then
        self.net:float()
    else
        self.net:cuda()
    end
end

function Flashlight:load_caffe_model(model, weights)
    self.net = loadcaffe.load(model, weights)
    self:remove_batch_norm(self.net)
    print(self.net)
    if self.backend == "cpu" then
        self.net:float()
    else
        self.net:cuda()
    end
end

function Flashlight:remove_batch_norm(net)
    for i, module in ipairs(net.modules) do
        if torch.type(module) == 'nn.SpatialBatchNormalization' then
            net:remove(i)
        end
    end
end

function Flashlight:predict(image)
    if self.backend == "cuda" then
        image = image:cuda()
    end
    self.net:evaluate()
    local output = self.net:forward(image)
    return output:float()
end

-- Retrieve the filter responses caused by passing the image through the model
-- Each table of filter responses from a layer has a field 'ADDED_NAME' 
-- added to it which contains the name of the layer type. This is to make
-- reviewing filter responses and mapping them back to layers easier...
-- Return the filter responses in a table 
function Flashlight:get_convolution_activation()
    self.filterResponses = {}
    for i, curModule in ipairs(self.net.modules) do
        curModule['ADDED_NAME'] = torch.type(curModule)
        if curModule.ADDED_NAME == "nn.SpatialConvolution" then
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
