require 'nn'
local TripletNetBatch, parent = torch.class('nn.TripletNetBatch', 'nn.Module')

local function SizeSquashed(x)
    local sz = x:size():totable()
    local first = table.remove(sz,1)
    sz[1] = sz[1]*first
    return torch.LongStorage(sz)
end

function TripletNetBatch:__init(net, dist)
    self.dist = dist or nn.PairwiseDistance(2)
    self.Net = net or nn.Sequential()
    self.output = torch.Tensor()
    self.netOutput = torch.Tensor()
    self.distGradInput = torch.Tensor()
end

function TripletNetBatch:add(module)
    self.Net:add(module)
end

function TripletNetBatch:updateOutput(input)

    local x = input:view(SizeSquashed(input))
    self.output:resize(input:size(2), input:size(1)- 1):typeAs(x)

    self.netOutput = self.Net:updateOutput(x)
    self.sub_output = torch.chunk(self.netOutput, input:size(1))
    
    for i=1, input:size(1)-1 do
        self.output[{{},i}]:copy(self.dist:updateOutput({self.sub_output[1],self.sub_output[i+1]}))
    end
    return self.output
end
function TripletNetBatch:updateGradInput(input,gradOutput)
    local sz = self.netOutput:size():totable()
    sz[1] = input:size(2)
    table.insert(sz,1,input:size(1))
    self.distGradInput:resize(unpack(sz))
    self.distGradInput:zero()
    for i=1, input:size(1)-1 do
        local dyi = gradOutput[{{},i}]
        local dEi = self.dist:updateGradInput({self.sub_output[1],self.sub_output[i+1]},dyi)
        self.distGradInput[1]:add(dEi[1])
        self.distGradInput[i+1]:copy(dEi[2])
    end

    local x = input:view(SizeSquashed(input))
    self.gradInput = self.Net:updateGradInput(x, self.distGradInput:view(SizeSquashed(self.distGradInput)))
    return self.gradInput
end

function TripletNetBatch:accGradParameters(input, gradOutput, scale)
    local x = input:view(SizeSquashed(input))
    self.Net:accGradParameters(x, self.distGradInput:view(SizeSquashed(self.distGradInput)), scale)
end


function TripletNetBatch:training()
    self.Net:training()
end

function TripletNetBatch:evaluate()
    self.Net:evaluate()
end

function TripletNetBatch:getParameters()
    local w, g = self.Net:getParameters()
    return w,g
end
function TripletNetBatch:parameters()
    return self.Net:parameters()
end

function TripletNetBatch:type(t)
    self.Net:type(t)
    self.dist:type(t)
    self.output = self.output:type(t)
    self.netOutput = self.netOutput:type(t)
    self.distGradInput = self.distGradInput:type(t)

end

