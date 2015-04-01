local opt = opt or {}
local Dataset = opt.dataset or 'SVHN'
local PreProcDir = opt.preProcDir or './'
local Whiten = opt.whiten or false
local DataPath = opt.datapath or '/home/ehoffer/Datasets/'
local SimpleNormalization = (opt.normalize==1) or false

local TestData
local TrainData
local Classes

if Dataset =='Cifar100' then
    TrainData = torch.load(DataPath .. 'Cifar100/cifar100-train.t7')
    TestData = torch.load(DataPath .. 'Cifar100/cifar100-test.t7')
    TrainData.labelCoarse:add(1)
    TestData.labelCoarse:add(1)
    Classes = torch.linspace(1,100,100):storage():totable()
elseif Dataset == 'Cifar10' then
    TrainData = torch.load(DataPath .. 'Cifar10/cifar10-train.t7')
    TestData = torch.load(DataPath .. 'Cifar10/cifar10-test.t7')
    Classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
elseif Dataset == 'STL10' then
    TrainData = torch.load(DataPath .. 'STL10/stl10-train.t7')
    TestData = torch.load(DataPath .. 'STL10/stl10-test.t7')
    Classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
    TestData.label = TestData.label:add(-1):byte()
    TrainData.label = TrainData.label:add(-1):byte()
elseif Dataset == 'MNIST' then
    TrainData = torch.load(DataPath .. 'MNIST/mnist-train.t7')
    TestData = torch.load(DataPath .. 'MNIST/mnist-test.t7')
    Classes = {1,2,3,4,5,6,7,8,9,0}
    TestData.data = TestData.data:view(TestData.data:size(1),1,28,28)
    TrainData.data = TrainData.data:view(TrainData.data:size(1),1,28,28)
    TestData.label = TestData.label:byte()
    TrainData.label = TrainData.label:byte()

elseif Dataset == 'SVHN' then
    TrainData = torch.load(DataPath .. 'SVHN/train_32x32.t7','ascii')
    ExtraData = torch.load(DataPath .. 'SVHN/extra_32x32.t7','ascii')
    TrainData.X = torch.cat(TrainData.X, ExtraData.X,1)
    TrainData.y = torch.cat(TrainData.y[1], ExtraData.y[1],1)
    TrainData = {data = TrainData.X, label = TrainData.y}
    TrainData.label = TrainData.label:add(-1):byte()
    TrainData.X = nil
    TrainData.y = nil
    ExtraData = nil

    TestData = torch.load(DataPath .. 'SVHN/test_32x32.t7','ascii')
    TestData = {data = TestData.X, label = TestData.y[1]}
    TestData.label = TestData.label:add(-1):byte()
    Classes = {1,2,3,4,5,6,7,8,9,0}
end
TrainData.label:add(1)
TestData.label:add(1)

--Preprocesss

TrainData.data = TrainData.data:float()
TestData.data = TestData.data:float()
local _, channels, y_size, x_size = unpack(TrainData.data:size():totable())
if SimpleNormalization then
    local mean = TrainData.data:mean()
    local std = TrainData.data:std()

    TrainData.data:add(-mean):div(std)
    TestData.data:add(-mean):div(std)
else
    --Preprocesss
    local meansfile = paths.concat(PreProcDir,'means.t7')
    if Whiten then
        require 'unsup'
        local means, P, invP
        local Pfile = paths.concat(PreProcDir,'P.t7')
        local invPfile = paths.concat(PreProcDir,'invP.t7')

        if (paths.filep(Pfile) and paths.filep(invPfile) and paths.filep(meansfile)) then
            P = torch.load(Pfile)
            invP = torch.load(invPfile)
            means = torch.load(meansfile)
            TrainData.data = unsup.zca_whiten(TrainData.data, means, P, invP)
        else
            TrainData.data, means, P, invP = unsup.zca_whiten(TrainData.data)
            torch.save(Pfile,P)
            torch.save(invPfile,invP)
            torch.save(meansfile,means)
        end
        TestData.data = unsup.zca_whiten(TestData.data, means, P, invP)


        TrainData.data = TrainData.data:float()
        TestData.data = TestData.data:float()

    else
        local means, std
        local loaded = false
        local stdfile = paths.concat(PreProcDir,'std.t7')
        if paths.filep(meansfile) and paths.filep(stdfile) then
            means = torch.load(meansfile)
            std = torch.load(stdfile)
            loaded = true
        end
        if not loaded then
            means = torch.mean(TrainData.data, 1):squeeze()
        end
        TrainData.data:add(-1, means:view(1,channels,y_size,x_size):expand(TrainData.data:size(1),channels,y_size,x_size))
        TestData.data:add(-1, means:view(1,channels,y_size,x_size):expand(TestData.data:size(1),channels,y_size,x_size))

        if not loaded then
            std = torch.std(TrainData.data, 1):squeeze()
        end
        TrainData.data:cdiv(std:view(1,channels,y_size,x_size):expand(TrainData.data:size(1),channels,y_size,x_size))
        TestData.data:cdiv(std:view(1,channels,y_size,x_size):expand(TestData.data:size(1),channels,y_size,x_size))

        if not loaded then
            torch.save(meansfile,means)
            torch.save(stdfile,std)
        end

    end
end

function ArrangeByLabel(labels)
    local numClasses = labels:max()
    local Ordered = {}

    for i=1,labels:size(1) do

        if Ordered[labels[i]] == nil then
            Ordered[labels[i]] = {}
        end
        table.insert(Ordered[labels[i]], i)
    end
    return Ordered
end

function GenerateList(labels,num, size)
    local list = torch.IntTensor(size,num)
    local Ordered = ArrangeByLabel(labels)
    local nClasses = #Ordered
    local c = torch.IntTensor(num-1)

    for i=1, size do

        c[1] = math.random(nClasses) --compared class
        local n1 = math.random(#Ordered[c[1]])
        local n_last = math.random(#Ordered[c[1]])
        
        while n_last == n1 do
            n_last = math.random(#Ordered[c[1]])
        end

        list[i][1] = Ordered[c[1]][n1]
        list[i][num] = Ordered[c[1]][n_last]

        for j=2,num-1 do --dissimilar classes
            repeat
                c[j] = math.random(nClasses)
            until c[j] ~= c[1]

            local n_j = math.random(#Ordered[c[j]])
            list[i][j] = Ordered[c[j]][n_j]
        end
    end
    return list

end

function GenerateListTriplets(labels, size)
    local list = torch.IntTensor(size,3)
    local Ordered = ArrangeByLabel(labels)
    local nClasses = #Ordered
    for i=1, size do
        local c1 = math.random(nClasses)
        local c2 = math.random(nClasses)
        while c2 == c1 do
            c2 = math.random(nClasses)
        end
        local n1 = math.random(#Ordered[c1])
        local n2 = math.random(#Ordered[c2])
        local n3 = math.random(#Ordered[c1])
        while n3 == n1 do
            n3 = math.random(#Ordered[c1])
        end

        list[i][1] = Ordered[c1][n1]
        list[i][2] = Ordered[c2][n2]
        list[i][3] = Ordered[c1][n3]
    end
    return list

end
function GenerateBiasedListTriplets(labels, size)
    local list = torch.IntTensor(size,3)
    local Ordered = ArrangeByLabel(labels)
    local nClasses = #Ordered
    for i=1, size do
        local c1 = math.random(nClasses)
        local c2 = math.random(nClasses)
        while c2 == c1 do
            c2 = math.random(nClasses)
        end
        local same_loc = torch.LongTensor(#Ordered[c1])
        local new_loc = torch.LongTensor(#Ordered[c2])
        for k=1,#Ordered[c1] do
            same_loc[k] = Ordered[c1][k]
        end
        for k=1,#Ordered[c2] do
            new_loc[k] = Ordered[c2][k]
        end

        local n1 = math.random(#Ordered[c1])
        list[i][1] = Ordered[c1][n1]
        local n3=n1
        while n1==n3 do
            n3 = math.random(#Ordered[c1])
        end

        local p2 = DistanceTensor[Ordered[c1][n3]]:index(1,new_loc):float()
        p2 = torch.cdiv(torch.ones(p2:nElement()):float(),p2)
        local n2 = dist.cat.rnd(p2)--math.random(#Ordered[c2])
        -- local p2 = DistanceTensor[list[i][1]]:index(1,new_loc):float()
        -- p2 = torch.cdiv(torch.ones(p2:nElement()):float(),p2)
        -- local n2 = dist.cat.rnd(p2)--math.random(#Ordered[c2])
        -- local p3 = DistanceTensor[list[i][1]]:index(1,same_loc):float()
        -- local n3 = dist.cat.rnd(p3)--math.random(#Ordered[c2])math.random(#Ordered[c1])
        n2 = n2[1]
        --n3 = n3[1]
        list[i][2] = Ordered[c2][n2]
        list[i][3] = Ordered[c1][n3]
    end
    return list

end


function hash_c(c1,c2)
    return (c1*10 + c2 -10)
end

function CreateDistanceTensor(data,labels, model)
    local Rep = ForwardModel(model,data)
    local Dist = torch.ByteTensor(data:size(1),data:size(1)):zero()
    for i=1,data:size(1) do
        for j=i+1,data:size(1) do
            Dist[i][j] = math.ceil(torch.dist(Rep[i],Rep[j]))
        end
    end
    return Dist
end
return{
    TrainData = TrainData,
    TestData = TestData,
    Classes = Classes

}
