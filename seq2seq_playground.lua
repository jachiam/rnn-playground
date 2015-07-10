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

	lr
	lambda
	gamma
	grad_clip
	clip_to

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

	-- RMSprop and clipping gradients
	s2s.lr = args.lr or 0.1		-- learning rate
	s2s.lambda = args.lambda or 0.4
	s2s.gamma = args.gamma or 0.4
	s2s.grad_clip = args.grad_clip or true
	s2s.clip_to = args.clip_to or 5

	-- Data
	if s2s.data_flag and not(args.rawdata) then
		print('Skipping data step; there is already data available...')
	else
		s2s.rawdata = args.rawdata or data()
		print('Creating embedding/deembedding tables for characters in data...')
		s2s.embed, s2s.deembed, s2s.numkeys = data_processing(s2s.rawdata)
		print('Finished making embed/deembed tables.') 

		print('Making tensor data and embedded data...')
		s2s.tensor_data = one_hot(s2s.rawdata,s2s.embed,s2s.deembed,s2s.numkeys)
		s2s.embedded_data = embed(s2s.rawdata,s2s.embed)
		print('Finished making tensor data and embedded data.')
		s2s.data_flag = true
	end

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

function data_processing(rawdata)
	local embeddings = {}
	local deembeddings = {}
	local numkeys = 0
	for char in rawdata:gmatch'.' do
		if not embeddings[char] then 
			embeddings[char] = numkeys + 1
			deembeddings[numkeys + 1] = char
			numkeys = numkeys + 1
		end
	end
	return embeddings, deembeddings, numkeys
end

function one_hot(rawdata,embeddings,deembeddings,numkeys)
	--local embeddings, deembeddings, numkeys = data_processing(rawdata)
	local tensor_data = torch.Tensor(rawdata:len(),numkeys):zero()
	local char
	for i=1,rawdata:len() do
		char = rawdata:sub(i,i)
		tensor_data[i][embeddings[char]] = 1
	end
	return tensor_data
end

function embed(rawdata,embeddings)
	local embedded_data = torch.Tensor(rawdata:len()):zero()
	for i=1,rawdata:len() do
		embedded_data[i] = embeddings[rawdata:sub(i,i)]
	end
	return embedded_data
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
	collectgarbage()
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
		collectgarbage()
	end
end

-- First index is time slice
-- Second is element-in-batch
function minibatch_loader()
	local i
	local X = torch.Tensor(s2s.seq_len,s2s.batch_size,s2s.numkeys):zero()
	local Y = torch.Tensor(s2s.seq_len,s2s.batch_size):zero()
	local H = torch.Tensor(s2s.seq_len,s2s.batch_size,s2s.n_hidden):zero()

	for n=1,s2s.batch_size do
		i = torch.ceil(torch.uniform()*(s2s.rawdata:len() - s2s.seq_len))
		X[{{},{n}}] = s2s.tensor_data[{{i,i + s2s.seq_len - 1},}]
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
