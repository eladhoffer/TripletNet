
require 'cudnn'

local model = nn.Sequential() 

-- Convolution Layers

model:add(cudnn.SpatialConvolution(3, 64, 5, 5 ))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.25))

model:add(cudnn.SpatialConvolution(64, 128, 3, 3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2))
model:add(nn.Dropout(0.25))

model:add(cudnn.SpatialConvolution(128, 256, 3, 3))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(2, 2))


model:add(nn.Dropout(0.25))
model:add(cudnn.SpatialConvolution(256, 128, 2,2))
model:add(nn.ReLU())
model:add(nn.View(128))


return model
