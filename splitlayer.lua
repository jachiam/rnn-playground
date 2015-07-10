require 'nn'
require 'cutorch'

--[[
Overview:
	This layer accepts a vector with N elements as an input,
	and splits that vector into k sub-vectors.

	Or, if the input is a batch of vectors, it splits each 
	vector in the batch accordingly.

	This returns a table with k elements. If the input was a
	vector, then output[j] has n_j elements, where

		sum_{j=1}^k n_j = N.

	If the input was a batch of M N-vectors, then output[j] has
	M x n_j elements. 

]]

local splitlayer = torch.class('nn.SplitLayer','nn.Module')

-- splitSizes should be a table of k elements,
-- {n_1, n_2, ..., n_k}. We must have
-- sum_i n_i = N. 
function splitlayer:__init(N,splitSizes)
	self.N = N
	self.splitSizes = splitSizes

	-- Check that splitsizes add up to N
	local m = 0
	for i=1,#self.splitSizes do m = m + self.splitSizes[i] end
	assert(m == self.N)
	self.output = {}
	self.gradInput = torch.Tensor()
end

function splitlayer:forward(x)
	local x = x

	local ptr = 1

	for j=1,#self.splitSizes do
		if x:dim() == 1 then
			self.output[j] = x[{{ptr, ptr+self.splitSizes[j] - 1}}]
		else
			self.output[j] = x[{{},{ptr, ptr+self.splitSizes[j] - 1}}]
		end
		ptr = ptr + self.splitSizes[j]
	end

	return self.output
end

function splitlayer:updateOutput(input)
	return self:forward(input)
end

function splitlayer:updateGradInput(input,gradOutput)
	self.gradInput = torch.Tensor():typeAs(input):resizeAs(input):zero()

	local ptr = 1
	for i=1,#self.splitSizes do
		if input:dim() == 1 then
			self.gradInput[{{ptr, ptr+self.splitSizes[i] - 1}}] = gradOutput[i]
		else
			self.gradInput[{{},{ptr, ptr+self.splitSizes[i] - 1}}] = gradOutput[i]
		end
		ptr = ptr + self.splitSizes[i]
	end

	return self.gradInput
end
