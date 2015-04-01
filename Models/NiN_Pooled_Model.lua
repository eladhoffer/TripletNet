
require 'cudnn'

local model = nn.Sequential() 


model:add(cudnn.SpatialConvolution(3, 192, 5, 5,1,1,2,2 ))
model:add(cudnn.ReLU(true))
model:add(nn.SpatialConvolutionMM(192, 160,1,1 ))
model:add(cudnn.ReLU(true))
model:add(nn.SpatialConvolutionMM(160,96, 1,1 ))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3, 3,2,2))
--model:add(nn.Dropout(0.5))
model:add(cudnn.SpatialConvolution(96,192, 5,5,1,1,2,2 ))
model:add(cudnn.ReLU(true))
model:add(nn.SpatialConvolutionMM(192,192, 1,1 ))
model:add(cudnn.ReLU(true))

model:add(nn.SpatialConvolutionMM(192,192, 1,1 ))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3,3,2,2))
--model:add(nn.Dropout(0.5))
model:add(cudnn.SpatialConvolution(192,192, 3,3 ,1,1,1,1))
model:add(cudnn.ReLU(true))
model:add(nn.SpatialConvolutionMM(192,192, 1,1 ))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(7,7))
model:add(nn.View(192))

return model
