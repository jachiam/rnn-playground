require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'LSTM'
local model_utils = require 'model_utils'

s2s = {}

--[[ Arguments:
	gpu
	batch_size
	seq_len
	collect_often

	lr
	lambda
	gamma
	grad_clip
	clip_to

	pattern
	lower
	rawdata

	layer_sizes
	peepholes
]]
function init(args)

	-- Verbosity
	s2s.report_freq = args.report_freq or 10

	-- Computational
	s2s.gpu = args.gpu or -1
	s2s.batch_size = args.batch_size or 50
	s2s.seq_len = args.seq_len or 50
	s2s.collect_often = args.collect_often or false

	-- RMSprop and clipping gradients
	s2s.lr = args.lr or 0.1		-- learning rate
	s2s.lambda = args.lambda or 0.4
	s2s.gamma = args.gamma or 0.4
	s2s.grad_clip = args.grad_clip or true
	s2s.clip_to = args.clip_to or 5

	-- Data
	s2s.pattern = args.pattern or '.'
	if s2s.pattern = 'word' then
		s2s.pattern = '%a+\'?%a+'
	end
	s2s.lower = args.lower or true
	s2s.rawdata = args.rawdata or data()
	print('Creating embedding/deembedding tables for characters in data...')
	s2s.embed, s2s.deembed, s2s.numkeys, s2s.numwords, s2s.embedded_data = data_processing(s2s.rawdata,s2s.pattern)
	s2s.eye = torch.eye(s2s.numkeys)
	print('Finished making embed/deembed tables.') 
	print('Finished making embedded data.')

	-- Networks and parameters
	s2s.layer_sizes = args.layer_sizes or {128}
	s2s.peepholes = args.peepholes or false
	s2s.n_hidden = 0
	for i=1,#s2s.layer_sizes do s2s.n_hidden = s2s.n_hidden + 2*s2s.layer_sizes[i] end

	print('Making LSTM...')
	protos = {}
	clones = {}
	protos.LSTM = LSTM(s2s.numkeys,s2s.layer_sizes,s2s.peepholes,true)
	if s2s.gpu >= 0 then protos.LSTM:cuda() end
	s2s.params, s2s.gradparams = protos.LSTM:getParameters()
	s2s.v = s2s.gradparams:clone():zero()
	print('Cloning LSTM...')
	clones.LSTM = model_utils.clone_many_times(protos.LSTM,s2s.seq_len)
	collectgarbage()
	print('LSTM is done.')

	print('Making criterion...')
	local criterion_input_1, criterion_input_2, criterion_out
	criterion_input_1 = nn.Identity()()
	criterion_input_2 = nn.Identity()()
	criterion_out = nn.ClassNLLCriterion()({nn.LogSoftMax()(criterion_input_1),criterion_input_2})
	protos.criterion = nn.gModule({criterion_input_1,criterion_input_2},{criterion_out})
	if s2s.gpu >= 0 then protos.criterion:cuda() end
	print('Cloning criterion...')
	clones.criterion = model_utils.clone_many_times(protos.criterion,s2s.seq_len)
	collectgarbage()
	print('Criterion is done.')
	
	if s2s.gpu >=0 then
		free,tot = cutorch.getMemoryUsage()
		print('Free fraction of memory remaining: ', free/tot)
	end
	print('Number of trainable parameters: ', s2s.params:nElement())

end


-- DATA MANIPULATION FUNCTIONS
function data()
	local f = torch.DiskFile('input.txt')
	local rawdata = f:readString('*a')
	f:close()
	return rawdata
end

function data_processing(rawdata,pattern)
	local embeddings = {}
	local deembeddings = {}
	local numkeys = 0
	local numwords = 0
	local breakapart = rawdata:gmatch(pattern)
	local token
	for char in breakapart do
		if s2s.lower then 
			token = char:lower()
		else
			token = char
		end
		numwords = numwords + 1
		if not embeddings[token] then 
			embeddings[token] = numkeys + 1
			deembeddings[numkeys + 1] = token
			numkeys = numkeys + 1
		end
	end

	local tokenized = torch.zeros(numwords)
	local i=1
	breakapart = rawdata:gmatch(pattern)
	for char in breakapart do
		if s2s.lower then 
			token = char:lower()
		else
			token = char
		end
		tokenized[i] = embeddings[token]
		i = i + 1
	end

	return embeddings, deembeddings, numkeys, numwords, tokenized
end


-- TRAINING

function fwd()
	total_loss = 0
	for i=1,s2s.seq_len do
		clones.LSTM[i]:forward({H[i],X[i]})
		if i < s2s.seq_len then H[i+1] = clones.LSTM[i].output[1] end
		loss = clones.criterion[i]:forward({clones.LSTM[i].output[2],Y[i]})
		total_loss = total_loss + loss[1]
	end
	if s2s.collect_often then collectgarbage() end
end

function bwd()
	for i=s2s.seq_len,1,-1 do
		clones.criterion[i]:backward({clones.LSTM[i].output[2],Y[i]},{1})
		if i < s2s.seq_len then
			clones.LSTM[i]:backward({H[i],X[i]},{
				clones.LSTM[i+1].gradInput[1],
				clones.criterion[i].gradInput[1]
				})
		else
			clones.LSTM[i]:backward({H[i],X[i]},{
				torch.Tensor():typeAs(H[i]):resizeAs(H[i]):zero(),
				clones.criterion[i].gradInput[1]
				})
		end
		if s2s.collect_often then collectgarbage() end
	end
end

-- First index is time slice
-- Second is element-in-batch
function minibatch_loader()
	local i
	local preX
	local X = torch.Tensor(s2s.seq_len,s2s.batch_size,s2s.numkeys):zero()
	local Y = torch.Tensor(s2s.seq_len,s2s.batch_size):zero()
	local H = torch.Tensor(s2s.seq_len,s2s.batch_size,s2s.n_hidden):zero()

	for n=1,s2s.batch_size do
		i = torch.ceil(torch.uniform()*(s2s.numwords - s2s.seq_len))
		preX = s2s.embedded_data[{{i,i + s2s.seq_len - 1}}]:long()
		X[{{},{n}}]:copy(s2s.eye:index(1,preX))
		Y[{{},{n}}] = s2s.embedded_data[{{i+1,i + s2s.seq_len}}]
	end

	if s2s.gpu >= 1 then
		X = X:float():cuda()
		Y = Y:float():cuda()
		H = H:float():cuda()
	end

	return X,Y,H
end


function train_network_one_step()
	X,Y,H = minibatch_loader()
	s2s.gradparams:zero()
	fwd()
	bwd()
	if s2s.grad_clip then
		s2s.gradparams:clamp(-s2s.clip_to, s2s.clip_to)
	end
	-- RMSprop:
	local grad = s2s.gradparams:clone()
	grad:pow(2):mul(1 - s2s.gamma)
	s2s.v:mul(s2s.gamma):add(grad)
	s2s.gradparams:cdiv(torch.sqrt(s2s.v):add(s2s.lambda))
	s2s.params:add(-s2s.lr,s2s.gradparams)
	collectgarbage()
end

function train_network_N_steps(N)
	running_total_loss = 0
	for n=1,N do
		train_network_one_step()
		running_total_loss = running_total_loss + total_loss/s2s.seq_len
		if n % s2s.report_freq == 0 then 
			print('Average Error: ',running_total_loss/s2s.report_freq) 
			running_total_loss = 0
		end
	end
end


function sample_from_network(length)
	sample_text = ''
	i = torch.ceil(torch.uniform()*s2s.numkeys)
	xcur = torch.Tensor(s2s.numkeys):zero()
	hcur = torch.Tensor(s2s.n_hidden):zero()
	softmax = nn.SoftMax()
	if s2s.gpu >= 1 then
		xcur = xcur:float():cuda()
		hcur = hcur:float():cuda()
		softmax:cuda()
	end
	xcur[i] = 1
	for n=1,length do
		sample_text = sample_text .. s2s.deembed[i]
		protos.LSTM:forward({hcur,xcur})
		hcur = protos.LSTM.output[1]
		pred = protos.LSTM.output[2]
		probs = softmax:forward(pred)
		probs:div(torch.sum(probs))
		next_char = torch.multinomial(probs:float(), 1):resize(1):float()
		i = next_char[1]
		xcur:zero()
		xcur[i] = 1
	end
	print('Sample text: ',sample_text)
end

function memory()
	free,tot = cutorch.getMemoryUsage()
	print(free/tot)
end
