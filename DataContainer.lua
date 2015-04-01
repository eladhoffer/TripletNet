require 'torch'
require 'dok'

require 'image'
local DataContainer = torch.class('DataContainer')

local function CatNumSize(num,size)
    local stg = torch.LongStorage(size:size()+1)
    stg[1] = num
    for i=2,stg:size() do
        stg[i]=size[i-1]
    end
    return stg
end
function DataContainer:__init(...)
    local args = dok.unpack(
    {...},
    'InitializeData',
    'Initializes a DataContainer ',
    {arg='BatchSize', type='number', help='Number of Elements in each Batch',req = true},
    {arg='TensorType', type='string', help='Type of Tensor', default = 'torch.FloatTensor'},
    {arg='ExtractFunction', type='function', help='function used to extract Data, Label and Info', default= function(...) return ... end},
    {arg='List', type='userdata', help='source of DataContainer', req=true},
    {arg='Data', type='userdata', help='Data', req = true},
    {arg='ListGenFunc', type='function', help='Generate new list'},
    {arg='Augment', type='boolean', help='augment data',default=false}
    )

    self.BatchSize = args.BatchSize
    self.TensorType = args.TensorType
    self.ExtractFunction = args.ExtractFunction
    self.Augment = args.Augment
    self.Batch = torch.Tensor():type(self.TensorType)
    self.Data = args.Data
    self.List = args.List
    self.ListGenFunc = args.ListGenFunc
    self.NumEachSet = self.List:size(2)
    self:Reset()
end

function DataContainer:Reset()
    self.CurrentItem = 1
end

function DataContainer:size()
    return self.List:size(1)
end

function DataContainer:Reset()
    self.CurrentItem = 1
end


function DataContainer:__tostring__()
    local str = 'DataContainer:\n'
    if self:size() > 0 then
        str = str .. ' + num samples : '.. self:size()
    else
        str = str .. ' + empty set...'
    end
    return str
end

function DataContainer:ShuffleItems()
    local RandOrder = torch.randperm(self.List:size(1)):long()
    self.List = self.List:indexCopy(1,RandOrder,self.List)
    print('(DataContainer)===>Shuffling Items')

end
function DataContainer:GenerateList()
    self.List = self.ListGenFunc()

end

function DataContainer:GetNextBatch()
    local size = math.min(self:size()-self.CurrentItem + 1, self.BatchSize )
    if size <= 0 then
        return nil
    end

    if self.Batch:dim() == 0 or size < self.BatchSize then
        local nsz = CatNumSize(self.NumEachSet, CatNumSize(size, self.Data[1]:size()))
        self.Batch:resize(nsz)
    end
    local batch_table = {}
    for i=1, self.NumEachSet do
        local d = self.Data:index(1,self.List[{{self.CurrentItem,self.CurrentItem+size-1},i}]:long())
        self.Batch[i]:copy(d)
    end
    local side = self.Data:size(3)
    if self.Augment then
        for l=1,self.NumEachSet do
            for i=1,size do
                local sz = math.random(side/4 + 1) - 1
                local hflip = math.random(2)==1

                local startx = math.random(sz) 
                local starty = math.random(sz) 
                local img = self.Batch[l][i]:narrow(2,starty,side-sz):narrow(3,startx,side-sz):float()
                if hflip then
                    img = image.hflip(img)
                end
                img = image.scale(img,side,side)
                self.Batch[l][i]:copy(img)
            end
        end
    end
    local list = self.List[{{self.CurrentItem,self.CurrentItem+size-1},i}]:long()
    self.CurrentItem = self.CurrentItem + size
    return self.Batch, list
end








