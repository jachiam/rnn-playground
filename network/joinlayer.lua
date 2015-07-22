require 'nn'

--[[
Overview:
	This layer accepts a table of vectors and combines them
	all into one larger vector.

	Or, if this input is a table of batches of vectors, it
	combines them into an appropriate single batch of vectors.

	If the input is a set of k vectors of dimension n_j, then
	the output is a vector with N = sum_{j=1}^k n_j entries.

	If the input is a set of k M x n_j batch vectors, then the 
	output is an M x (sum_{j=1}^k n_j) tensor.

	Table entries should be numbered from 1 to k.
]]

local joinlayer = torch.class('nn.JoinLayer','nn.Module')

function joinlayer:__init()
	self.gradInput = {}
	self.splitSizes = {}
	self.batch = false
end


function joinlayer:updateOutput(input)
	local X = input
	local splitSizes = {}

	-- Check the first entry in the table. If dim = 1, then all 
	-- other entries should also be vectors. Otherwise, all other
	-- entries should be batches.
	local batch = X[1]:dim() > 1
	local M
	if batch then M = X[1]:size(1) end
	-- save 'batch' for gradients later
	self.batch = batch

	local N = 0
	-- Check to make sure that either all table entries are
	-- vectors OR that all table entries are M-batches of vectors.
	for i=1,#X do
		if batch then
			assert(X[i]:size(1) == M)
			splitSizes[i] = X[i]:size(2)
			N = N + X[i]:size(2)
		else
			assert(X[i]:dim() == 1)
			splitSizes[i] = X[i]:size(1)
			N = N + X[i]:size(1)
		end
	end
	-- save the splitSizes for gradients later
	self.splitSizes = splitSizes

	-- Make the output have the appropriate size and type
	local size
	if batch then
		size = {M,N}
	else
		size = {N}
	end
	self.output = torch.Tensor(unpack(size)):typeAs(X[1])

	-- Build the output
	local ptr = 1
	for i=1,#X do
		if batch then
			self.output[{{},{ptr, ptr + splitSizes[i] - 1}}] = X[i]
		else
			self.output[{{ptr, ptr + splitSizes[i] - 1}}] = X[i]
		end
		ptr = ptr + splitSizes[i]
	end

	return self.output
end

function joinlayer:updateGradInput(input,gradOutput)

	self.gradInput = {}
	local ptr = 1
	for i=1,#self.splitSizes do
		if self.batch then
			self.gradInput[i] = gradOutput[{{},{ptr, ptr + self.splitSizes[i] - 1} }]
		else
			self.gradInput[i] = gradOutput[{{ptr, ptr + self.splitSizes[i] - 1} }]
		end
		ptr = ptr + self.splitSizes[i]
	end
	
	return self.gradInput
end
