require 'nn'
require 'cutorch'

local phlayer = torch.class('nn.PeepLayer','nn.Module')

function phlayer:__init(inputSize)
	self.inputSize = inputSize
	self.weight = torch.Tensor(1,inputSize)
	self.gradWeight = torch.Tensor(1,inputSize)
	self.gradInput = torch.Tensor()
	self.output = torch.Tensor()

	self:reset()
end

function phlayer:reset(stdv)
	if stdv then
		stdv = stdv * math.sqrt(3)
	else
		stdv = 1./math.sqrt(self.inputSize)
	end
	self.weight:uniform(-stdv,stdv)
	return self
end

function phlayer:updateOutput(input)
	if input:dim() == 1 then
		self.output = torch.cmul(self.weight,input)
	elseif input:dim() == 2 then
		local w = self.weight:expand(input:size()[1],self.inputSize)
		self.output = torch.cmul(w,input)		
	else
		error('input must be vector or matrix')
	end

	return self.output
end

function phlayer:updateGradInput(input,gradOutput)

	if input:dim() == 1 then
		self.gradInput = torch.cmul(self.weight,gradOutput)
	elseif input:dim() == 2 then
		local w = self.weight:expand(input:size()[1],self.inputSize)
		self.gradInput = torch.cmul(w,gradOutput)		
	else
		error('input must be vector or matrix')
	end
	

	return self.gradInput
end

function phlayer:accGradParameters(input,gradOutput,scale)
	scale = scale or 1

	self.gradWeight = torch.cmul(input,gradOutput)	
end
