require 'DataContainer'
require 'TripletNet'
require 'cutorch'
require 'eladtools'
require 'optim'
require 'xlua'
require 'trepl'
require 'DistanceRatioCriterion'
require 'cunn'
----------------------------------------------------------------------


cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a Triplet network on CIFAR 10/100')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model And Training Regime')
cmd:option('-modelsFolder',       './Models/',            'Models Folder')
cmd:option('-network',            'Model.lua',            'embedding network file - must return valid network.')
cmd:option('-LR',                 0.1,                    'learning rate')
cmd:option('-LRDecay',            1e-6,                   'learning rate decay (in # samples)')
cmd:option('-weightDecay',        1e-4,                   'L2 penalty on the weights')
cmd:option('-momentum',           0.9,                    'momentum')
cmd:option('-batchSize',          128,                    'batch size')
cmd:option('-optimization',       'sgd',                  'optimization method')
cmd:option('-epoch',              -1,                     'number of epochs to train, -1 for unbounded')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                      'number of threads')
cmd:option('-type',               'cuda',                 'float or cuda')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                     'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')

cmd:text('===>Data Options')
cmd:option('-dataset',            'Cifar10',              'Dataset - Cifar10 or Cifar100')
cmd:option('-size',               640000,                 'size of training list' )
cmd:option('-normalize',          1,                      '1 - normalize using only 1 mean and std values')
cmd:option('-whiten',             false,                  'whiten data')
cmd:option('-augment',            false,                  'Augment training data')
cmd:option('-preProcDir',         './PreProcData/',       'Data for pre-processing (means,P,invP)')

cmd:text('===>Misc')
cmd:option('-visualize',          false,                  'display first level filters after each epoch')


opt = cmd:parse(arg or {})
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)

opt.network = opt.modelsFolder .. paths.basename(opt.network, '.lua')
opt.save = paths.concat('./Results', opt.save)
opt.preProcDir = paths.concat(opt.preProcDir, opt.dataset .. '/')
os.execute('mkdir -p ' .. opt.preProcDir)
if opt.augment then
    require 'image'
end

----------------------------------------------------------------------
-- Model + Loss:

local EmbeddingNet = require(opt.network)
local TripletNet = nn.TripletNet(EmbeddingNet)
local Loss = nn.DistanceRatioCriterion()
TripletNet:cuda()
Loss:cuda()


local Weights, Gradients = TripletNet:getParameters()

if paths.filep(opt.load) then
    local w = torch.load(opt.load)
    print('Loaded')
    Weights:copy(w)
end

--TripletNet:RebuildNet() --if using TripletNet instead of TripletNetBatch

local data = require 'Data'
local SizeTrain = opt.size or 640000
local SizeTest = SizeTrain*0.1

function ReGenerateTrain()
    return GenerateList(data.TrainData.label,3, SizeTrain)
end
local TrainList = ReGenerateTrain()
local TestList = GenerateList(data.TestData.label,3, SizeTest)


------------------------- Output files configuration -----------------
os.execute('mkdir -p ' .. opt.save)
os.execute('cp ' .. opt.network .. '.lua ' .. opt.save)
cmd:log(opt.save .. '/Log.txt', opt)
local weights_filename = paths.concat(opt.save, 'Weights.t7')
local log_filename = paths.concat(opt.save,'ErrorProgress')
local Log = optim.Logger(log_filename)
----------------------------------------------------------------------

print '==> Embedding Network'
print(EmbeddingNet)
print '==> Triplet Network'
print(TripletNet)
print '==> Loss'
print(Loss)

----------------------------------------------------------------------
local TrainDataContainer = DataContainer{
    Data = data.TrainData.data,
    List = TrainList,
    TensorType = 'torch.CudaTensor',
    BatchSize = opt.batchSize,
    Augment = opt.augment,
    ListGenFunc = ReGenerateTrain
}

local TestDataContainer = DataContainer{
    Data = data.TestData.data,
    List = TestList,
    TensorType = 'torch.CudaTensor',
    BatchSize = opt.batchSize
}


local function ErrorCount(y)
    if torch.type(y) == 'table' then
      y = y[#y]
    end
    return (y[{{},2}]:ge(y[{{},1}]):sum())
end

local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}


local optimizer = Optimizer{
    Model = TripletNet,
    Loss = Loss,
    OptFunction = _G.optim[opt.optimization],
    OptState = optimState,
    Parameters = {Weights, Gradients},
}

function Train(DataC)
    DataC:Reset()
    DataC:GenerateList()
    TripletNet:training()
    local err = 0
    local num = 1
    local x = DataC:GetNextBatch()

    while x do
        local y = optimizer:optimize({x[1],x[2],x[3]}, 1)

        err = err + ErrorCount(y)
        xlua.progress(num*opt.batchSize, DataC:size())
        num = num + 1
        x = DataC:GetNextBatch()

    end
    return (err/DataC:size())
end

function Test(DataC)
    DataC:Reset()
    TripletNet:evaluate()
    local err = 0
    local x = DataC:GetNextBatch()
    local num = 1
    while x do
        local y = TripletNet:forward({x[1],x[2],x[3]})
        err = err + ErrorCount(y)
        xlua.progress(num*opt.batchSize, DataC:size())
        num = num +1
        x = DataC:GetNextBatch()
    end
    return (err/DataC:size())
end


local epoch = 1
print '\n==> Starting Training\n'
while epoch ~= opt.epoch do
    print('Epoch ' .. epoch)
    local ErrTrain = Train(TrainDataContainer)
    torch.save(weights_filename, Weights)
    print('Training Error = ' .. ErrTrain)
    local ErrTest = Test(TestDataContainer)
    print('Test Error = ' .. ErrTest)
    Log:add{['Training Error']= ErrTrain* 100, ['Test Error'] = ErrTest* 100}
    Log:style{['Training Error'] = '-', ['Test Error'] = '-'}
    Log:plot()

    if opt.visualize then
        require 'image'
        local weights = EmbeddingNet:get(1).weight:clone()
        --win = image.display(weights,5,nil,nil,nil,win)
        image.saveJPG(paths.concat(opt.save,'Filters_epoch'.. epoch .. '.jpg'), image.toDisplayTensor(weights))
    end

    epoch = epoch+1
end
