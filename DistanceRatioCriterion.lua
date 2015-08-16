require 'nn'
local DistanceRatioCriterion, parent = torch.class('nn.DistanceRatioCriterion', 'nn.Criterion')

function DistanceRatioCriterion:__init()
    parent.__init(self)
    self.SoftMax = nn.SoftMax()
    self.MSE = nn.MSECriterion()
    self.Target = torch.Tensor()
end

function DistanceRatioCriterion:createTarget(input, target)
    local target = target or 1
    self.Target:resizeAs(input):typeAs(input):zero()
    self.Target[{{},target}]:add(1)

end

function DistanceRatioCriterion:updateOutput(input, target)
    if not self.Target:isSameSizeAs(input) then
        self:createTarget(input, target)
    end
    self.output = self.MSE:updateOutput(self.SoftMax:updateOutput(input),self.Target)
    return self.output
end

function DistanceRatioCriterion:updateGradInput(input, target)
    if not self.Target:isSameSizeAs(input) then
        self:createTarget(input, target)
    end

    self.gradInput = self.SoftMax:updateGradInput(input, self.MSE:updateGradInput(self.SoftMax.output,self.Target))
    return self.gradInput
end

function DistanceRatioCriterion:type(t)
    parent.type(self, t)
    self.SoftMax:type(t)
    self.MSE:type(t)
    self.Target = self.Target:type(t)
    return self
end
