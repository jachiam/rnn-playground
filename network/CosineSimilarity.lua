require 'nn'

local cosines = torch.class('nn.CosineSimilarity','nn.Module')

function cosines:__init(divbyx)
	self.divbyx = divbyx or false
	self.output = torch.Tensor()
	self.gradInput = {}

	-- Useful things from forward pass to keep
	self.wx = torch.Tensor()
	self.wc = torch.Tensor()
	self.w = torch.Tensor()
	self.xc = torch.Tensor()
	self.x = torch.Tensor()
end

function cosines:updateOutput(input)
	-- W is an 'nkeys x embveclen' matrix
	-- X is an 'nbatch x embveclen' matrix
	-- the goal is to output a matrix C that is 'nbatch x nkeys',
	-- where C[i,j] is the cosine similarity of the vector X[i] with
	-- key embedding W[j]. That is, C[i,j] = dot(W[j],X[i]) / |W[j]|.
	local W, X = input[1], input[2]

	if W:dim() == 1 then W:resize(1,W:size(1)) end
	if X:dim() == 1 then X:resize(1,X:size(1)) end

	local w = W:clone()
	w = w:cmul(w):sum(2)
	self.wc = w:sqrt()
	w = self.wc:clone():t()		-- This is a vector of the norms |W[j]|

	self.wx = torch.mm(X,W:t())
	w = w:expandAs(self.wx)
	self.w = w

	self.output = torch.cdiv(self.wx,self.w)

	local x
	if self.divbyx then
		x = X:clone()
		x = x:cmul(x):sum(2)
		self.xc = x:sqrt()
		x = self.xc:clone()
		x = x:expandAs(self.wx)
		self.x = x
		self.output:cdiv(x)
	end

	return self.output
end

function cosines:updateGradInput(input,gradOutput)
	local W,X = input[1],input[2]

	if W:dim() == 1 then W:resize(1,W:size(1)) end
	if X:dim() == 1 then X:resize(1,X:size(1)) end

	-- oh man. OH MAN. are you ready for a bunch of ugly matrix ops?
	-- because i know i am.

	-- so here is how we are running this show. think of the forward
	-- as four major computational blocks.
	-- block 1: go from X to x, where x is the 'nbatch x nkeys' matrix
	-- with elements x[i,j] = |X[i,1:embveclen]|
	-- block 2: go from W to w, where w is the 'nbatch x nkeys' matrix 
	-- with elements w[i,j] = |W[j,1:embveclen]|
	-- block 3: take X and W and produce wx = XW^T
	-- block 4: take wx,w,x, and give q = wx/w/x. 

	-- but we are only using block 1 if divbyx. otherwise no block 1, and
	-- block 4 gives q = wx/w. 

	-- we are going to go graphways backwards through that. 
	-- hold on to your butts.

	-- so, the output from the module is q, which is 'nbatch x nkeys'
	-- gradOuput is therefore also 'nbatch x nkeys'

	-- first we will get the gradients of q with resp to wx,w,x.

	-- block 4 backward pass
	local g4x, g4w, g4wx
	g4wx = torch.ones(self.output:size()):typeAs(W):cdiv(self.w)
	g4w = -torch.cdiv(self.output,self.w)
	if self.divbyx then 
		g4wx = g4wx:cdiv(self.x) 
		g4x = -torch.cdiv(self.output,self.x)
	end

	-- now we make the gradients of wx,w,x with respect to the loss
	local glx, glw, glwx
	glwx = torch.cmul(gradOutput,g4wx)
	glw = torch.cmul(gradOutput,g4w)
	if self.divbyx then
		glx = torch.cmul(gradOutput,g4x)
	end

	-- optional block 1 backward pass.
	-- Things that happen in block 1:
	-- in       -a-        -b-        -c-        -d-
	-- X --> X cmul X --> sum(2) --> sqrt() --> expand
	local g1a,g1b,g1c,g1d
	if self.divbyx then
		-- First, backwards through the expand:
		g1d = glx:sum(2)
		-- Next, backwards through the sqrt:
		-- for y=sqrt(x), dy/dx = (1/2)(1/y)
		g1c = g1d:cmul(self.xc:clone():pow(-1)):mul(0.5)
		-- Backwards through the sum:
		g1b = g1c:expandAs(X)
		-- Backwards through the squaring:
		-- for y = x^2, dy/dx = 2x
		g1a = torch.cmul(g1b,X):mul(2)
	end

	-- block 2 backward pass.
	-- Things that happen in block 2:
	-- in       -a-        -b-        -c-        -d-
	-- W --> W cmul W --> sum(2) --> sqrt() --> expand
	local g2a,g2b,g2c,g2d
	-- First, backwards through the expand:
	g2d = glw:sum(1)
	-- Next, backwards through the sqrt:
	-- for y=sqrt(x), dy/dx = (1/2)(1/y)
	g2c = g2d:cmul(self.wc:clone():pow(-1)):mul(0.5):t()
	-- Backwards through the sum:
	g2b = g2c:expandAs(W)
	-- Backwards through the squaring:
	-- for y = x^2, dy/dx = 2x
	g2a = torch.cmul(g2b,W):mul(2)

	-- block 3 backward pass.
	-- Things that happen in block 2:
	-- in       out
	-- W,X --> XW^T
	local g3X, g3W
	g3X = torch.mm(glwx,W)
	g3W = torch.mm(glwx:t(),X)

	-- Sum up for final grads
	local gW,gX
	gW = g3W + g2a
	gX = g3X
	if self.divbyx then gX = gX + g1a end


	-- This was done by hand, and so, uh, you know. Who knows if it will work!
	self.gradInput = {gW,gX}
	return self.gradInput
end
