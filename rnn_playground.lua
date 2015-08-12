require 'torch'
require 'nn'
require 'nngraph'
require 'data_processing'
local model_utils = require 'model_utils'
package.path = package.path .. ";./network/?.lua"

xrnn = {}

--[[ Arguments:

	-------------------
	--   VERBOSITY   --
	-------------------
	report_freq	-- reports performance during training this often
	conv_verbose	-- if true, when conversing, if input symbol is unknown, declare so

	-------------------
	-- COMPUTATIONAL --
	-------------------
	gpu
	batch_size
	seq_len
	collect_often
	noclones	-- if true, gradients are calculated more slowly but much more memory-efficiently (default false)
			   (useful for big networks on small graphics cards)

	-------------------
	--  OPTIMIZATION --
	-------------------
	lr		-- learning rate
	lambda		-- RMSprop parameter
	gamma		-- RMSprop parameter
	grad_clip	-- boolean. clip gradients? yes or no. (currently always true)
	clip_to		-- clip gradients to what? (default 5)

	training_mode	-- if 1, randomly samples minibatches from anywhere in text corpus (default)
			   if 2, proceeds linearly through text corpus

	-------------------
	--     DATA      --
	-------------------
	pattern		-- lua pattern as string, or 'word' (experimental!). (defaults to '.', which matches all chars.)
	lower		-- processes all text in lowercase (default false)
	rawdata		-- if none given, rawdata will be obtained from input file
	usemostcommon	-- use only the most common pattern-matching entries in rawdata as input symbols (default false)
	useNmostcommon  -- how many?
	replace		-- some common replacements for word mode (it's to it is, etc.)
	filename	-- name of file containing training data rawtext (if not supplied, defaults to 'input.txt')

	-------------------
	--     MODEL     --
	-------------------
	layer_sizes	-- table containing sizes of layers (default {128})
	peepholes	-- for LSTM networks (default false)
	nl_type		-- for Vanilla RNN networks (default 'tanh')
	RNN_type	-- string, determines RNN type. options are 'LSTM', 'GRU', 'Vanilla'. (default 'LSTM')

]]
function init(args)

	-- Verbosity
	xrnn.report_freq = args.report_freq or 10
	xrnn.conv_verbose = args.conv_verbose or false

	-- Computational
	xrnn.gpu = args.gpu or -1
	if xrnn.gpu >= 0 then
		require 'cutorch'
		require 'cunn'
	end
	xrnn.batch_size = args.batch_size or 30
	xrnn.seq_len = args.seq_len or 50
	xrnn.collect_often = args.collect_often or false
	xrnn.noclones = args.noclones or false

	-- RMSprop and clipping gradients
	xrnn.lr = args.lr or 0.001		-- learning rate
	xrnn.lambda = args.lambda or 1e-8
	xrnn.gamma = args.gamma or 0.95
	xrnn.grad_clip = args.grad_clip or true
	xrnn.clip_to = args.clip_to or 5

	-- Training
	xrnn.training_mode = args.training_mode or 1

	-- Data
	xrnn.pattern = args.pattern or '.'
	xrnn.lower = args.lower or false
	xrnn.usemostcommon = args.usemostcommon or false
	xrnn.useNmostcommon = args.useNmostcommon or 4500
	xrnn.replace = args.replace or false
	xrnn.wordopt = {usemostcommon = xrnn.usemostcommon, useNmostcommon = xrnn.useNmostcommon, replace = xrnn.replace}
	xrnn.filename = args.filename or 'input.txt'
	xrnn.rawdata = args.rawdata or data(xrnn.filename)
	print('Creating embedding/deembedding tables for characters in data...')
	xrnn.embed, xrnn.deembed, xrnn.numkeys, xrnn.numwords, xrnn.tokenized, xrnn.freq_data = data_processing(xrnn.rawdata,xrnn.pattern,xrnn.lower,xrnn.wordopt)
	xrnn.numkeys = xrnn.numkeys + 1 		-- for unknown character
	print('Finished making embed/deembed tables.') 
	print('Finished making embedded data.')
	print('Dictionary has this many keys in it: ',xrnn.numkeys)

	-- Input mode things
	xrnn.eye = torch.eye(xrnn.numkeys)
	xrnn.decoder = true
	xrnn.rnn_input_size = xrnn.numkeys

	-- Networks and parameters
	xrnn.layer_sizes = args.layer_sizes or {128}
	xrnn.peepholes = args.peepholes or false
	xrnn.nl_type = args.nl_type or 'tanh'
	xrnn.n_hidden = 0

	xrnn.RNN_type = args.RNN_type or 'LSTM'

	-- Interpret RNN_type options and compute length of hidden state vector
	if xrnn.RNN_type == 'LSTM' then
		print('Making LSTM...')
		for i=1,#xrnn.layer_sizes do xrnn.n_hidden = xrnn.n_hidden + 2*xrnn.layer_sizes[i] end
		require 'LSTM'
	elseif xrnn.RNN_type == 'GRU' then
		print('Making GRU...')
		for i=1,#xrnn.layer_sizes do xrnn.n_hidden = xrnn.n_hidden + xrnn.layer_sizes[i] end
		require 'GRU'
	elseif xrnn.RNN_type == 'Vanilla' then
		print('Making Vanilla RNN...')
		for i=1,#xrnn.layer_sizes do xrnn.n_hidden = xrnn.n_hidden + xrnn.layer_sizes[i] end
		require 'VanillaRNN'
	end

	-- Build RNN and make references to its parameters and gradient
	if args.RNN then
		xrnn.RNN = args.RNN
	elseif xrnn.RNN_type == 'LSTM' then
		xrnn.RNN = LSTM(xrnn.rnn_input_size,xrnn.layer_sizes,{peepholes=xrnn.peepholes,decoder=xrnn.decoder,fgate_init=true})
	elseif xrnn.RNN_type == 'GRU' then
		xrnn.RNN = GRU(xrnn.rnn_input_size,xrnn.layer_sizes,{decoder=xrnn.decoder})
	elseif xrnn.RNN_type == 'Vanilla' then
		xrnn.RNN = VanillaRNN(xrnn.rnn_input_size,xrnn.layer_sizes,{decoder=xrnn.decoder,nl_type=xrnn.nl_type})
	end
	if xrnn.gpu >= 0 then xrnn.RNN:cuda() end
	xrnn.params, xrnn.gradparams = xrnn.RNN:getParameters()
	xrnn.v = xrnn.gradparams:clone():zero()
	print('RNN is done.')

	-- Make criterion
	print('Making criterion...')
	local criterion_input_1, criterion_input_2, criterion_out
	criterion_input_1 = nn.Identity()()
	criterion_input_2 = nn.Identity()()
	criterion_out = nn.ClassNLLCriterion()({nn.LogSoftMax()(criterion_input_1),criterion_input_2})
	xrnn.criterion = nn.gModule({criterion_input_1,criterion_input_2},{criterion_out})
	if xrnn.gpu >= 0 then xrnn.criterion:cuda() end
	print('Criterion is done.')

	-- Make RNN/criterion clones, if applicable
	if not(xrnn.noclones) then
		clones = {}
		print('Cloning RNN...')
		clones.RNN = model_utils.clone_many_times(xrnn.RNN,xrnn.seq_len)
		collectgarbage()
		print('Cloning criterion...')
		clones.criterion = model_utils.clone_many_times(xrnn.criterion,xrnn.seq_len)
		collectgarbage()
		print('Clones are done.')
	end
	
	if xrnn.gpu >=0 then
		free,tot = cutorch.getMemoryUsage()
		print('Free fraction of memory remaining: ', free/tot)
	end
	print('Number of trainable parameters: ', xrnn.params:nElement())

end


-- TRAINING

function grad_pass_with_clones()
	total_loss = 0
	for i=1,xrnn.seq_len do
		clones.RNN[i]:forward({H[i],X[i]})
		if i < xrnn.seq_len then H[i+1] = clones.RNN[i].output[1] end
		loss = clones.criterion[i]:forward({clones.RNN[i].output[2],Y[i]})
		total_loss = total_loss + loss[1]
	end
	if xrnn.collect_often then collectgarbage() end

	local gradH
	for i=xrnn.seq_len,1,-1 do
		clones.criterion[i]:backward({clones.RNN[i].output[2],Y[i]},{1})
		if i < xrnn.seq_len then
			gradH = clones.RNN[i+1].gradInput[1]
		else
			gradH = torch.Tensor():typeAs(H[i]):resizeAs(H[i]):zero()
		end
		clones.RNN[i]:backward({H[i],X[i]},{gradH,clones.criterion[i].gradInput[1]})
		if xrnn.collect_often then collectgarbage() end
	end
end

function grad_pass_no_clones()
	total_loss = 0

	outputs = {}
	-- fwd pass to get the outputs and hidden states
	for i=1,xrnn.seq_len do
		xrnn.RNN:forward({H[i],X[i]})
		if i < xrnn.seq_len then H[i+1] = xrnn.RNN.output[1] end
		outputs[i] = xrnn.RNN.output[2]
	end
	if xrnn.collect_often then collectgarbage() end

	gradInputs = {}
	-- bwd pass
	for i=xrnn.seq_len,1,-1 do
		loss = xrnn.criterion:forward({outputs[i],Y[i]})
		total_loss = total_loss + loss[1]
		xrnn.criterion:backward({outputs[i],Y[i]},{1})
		xrnn.RNN:forward({H[i],X[i]})
		if i < xrnn.seq_len then
			xrnn.RNN:backward({H[i],X[i]},{gradInputs[i], xrnn.criterion.gradInput[1]})
		else
			xrnn.RNN:backward({H[i],X[i]},{torch.Tensor():typeAs(H[i]):resizeAs(H[i]):zero(),xrnn.criterion.gradInput[1]})
		end
		gradInputs[i-1] = xrnn.RNN.gradInput[1]
	end
	if xrnn.collect_often then collectgarbage() end
end

-- First index is time slice
-- Second is element-in-batch
function minibatch_loader()
	local i
	local preX,postX
	local I = torch.Tensor(xrnn.seq_len,xrnn.batch_size):zero():long()
	local X = torch.Tensor(xrnn.seq_len,xrnn.batch_size,xrnn.rnn_input_size):zero()
	local Y = torch.Tensor(xrnn.seq_len,xrnn.batch_size):zero()
	local H = torch.Tensor(xrnn.seq_len,xrnn.batch_size,xrnn.n_hidden):zero()

	if xrnn.training_mode == 2 then
		if not(xrnn.pos_in_text) then
			xrnn.pos_in_text = 1
		end
	end

	for n=1,xrnn.batch_size do
		if xrnn.training_mode == 1 then
			i = torch.ceil(torch.uniform()*(xrnn.numwords - xrnn.seq_len))
		else
			i = xrnn.pos_in_text
		end
		preX = xrnn.tokenized[{{i,i + xrnn.seq_len - 1}}]:long()
		postX = xrnn.eye:index(1,preX)
		I[{{},{n}}]:copy(preX)
		X[{{},{n}}]:copy(postX)
		Y[{{},{n}}] = xrnn.tokenized[{{i+1,i + xrnn.seq_len}}]
		if xrnn.training_mode == 2 then
			xrnn.pos_in_text = xrnn.pos_in_text + xrnn.seq_len
			if xrnn.pos_in_text > xrnn.numwords - xrnn.seq_len then
				xrnn.pos_in_text = 1
			end
		end
	end

	if xrnn.gpu >= 0 then
		X = X:float():cuda()
		Y = Y:float():cuda()
		H = H:float():cuda()
	end

	return X,Y,H,I
end


function train_network_one_step()
	X,Y,H,I = minibatch_loader()
	xrnn.gradparams:zero()

	if xrnn.noclones then
		grad_pass_no_clones()
	else
		grad_pass_with_clones()
	end

	-- Average over batch and sequence length
	xrnn.gradparams:div(xrnn.batch_size):div(xrnn.seq_len)

	if xrnn.grad_clip then
		xrnn.gradparams:clamp(-xrnn.clip_to, xrnn.clip_to)
	end
	-- RMSprop:
	local grad = xrnn.gradparams:clone()
	grad:pow(2):mul(1 - xrnn.gamma)
	xrnn.v:mul(xrnn.gamma):add(grad)
	xrnn.gradparams:cdiv(torch.sqrt(xrnn.v):add(xrnn.lambda))
	xrnn.params:add(-xrnn.lr,xrnn.gradparams)

	collectgarbage()
end

function train_network_N_steps(N)
	running_total_loss = 0
	for n=1,N do
		train_network_one_step()
		if n==1 then init_error = total_loss/xrnn.seq_len end
		if total_loss/xrnn.seq_len > 3*init_error then
			print('Error is exploding. Current error: ', total_loss/xrnn.seq_len)
			print('Terminating training here.')
			break
		end
		running_total_loss = running_total_loss + total_loss/xrnn.seq_len
		if n % xrnn.report_freq == 0 then 
			if xrnn.training_mode == 1 then
				print('Average Error: ',running_total_loss/xrnn.report_freq,'Num Steps: ',n)
			else
				print('Average Error: ',running_total_loss/xrnn.report_freq,'Num Steps: ',n,' % thru text: ', round(100*xrnn.pos_in_text/xrnn.numwords))
			end 
			running_total_loss = 0
		end
	end
end

function epochs(N)
	local steps_per_epoch = xrnn.numwords/xrnn.seq_len/xrnn.batch_size
	xrnn.training_mode = 2
	for k=1,N do
		train_network_N_steps(steps_per_epoch)
	end
end

function round(x)
	return math.floor(x * 1000)/1000
end

-- SAMPLING

function tokenize_string(text)

	local splitstring = split_string(text,xrnn.pattern,xrnn.lower,xrnn.wordopt)
	local numtokens = #splitstring
	local tokenized = torch.zeros(numtokens)
	for i=1,numtokens do
		tokenized[i] = xrnn.embed[splitstring[i]] or xrnn.numkeys
		if tokenized[i] == xrnn.numkeys and xrnn.conv_verbose then
			print('Machine Subconscious: I did not recognize the word ' .. splitstring[i] .. '.')
			print()
		end
	end

	return tokenized:long(), numtokens
end

function string_to_rnn_input(text)
	local tokenized, numtokens = tokenize_string(text)
	local X = torch.zeros(numtokens,xrnn.rnn_input_size)
	X:copy(xrnn.eye:index(1,tokenized))
	return X
end

function sample_from_network(args)
	local X, xcur, hcur
	local length = args.length
	local chatmode = args.chatmode or false
	local toscreen = args.toscreen or not(chatmode)
	local primetext = args.primetext
	sample_text = ''

	xcur = torch.Tensor(xrnn.rnn_input_size):zero()
	hcur = args.hcur or torch.Tensor(xrnn.n_hidden):zero()

	softmax = nn.SoftMax()
	if xrnn.gpu >= 1 then
		xcur = xcur:float():cuda()
		hcur = hcur:float():cuda()
		softmax:cuda()
	end

	if not(primetext) then
		i = torch.ceil(torch.uniform()*xrnn.numkeys)
		xcur[i] = 1
	else
		X = string_to_rnn_input(primetext)
		if xrnn.gpu >=1 then
			X = X:cuda()
		end
		for n=1,X:size(1)-1 do
			xrnn.RNN:forward({hcur,X[n]})
			hcur = xrnn.RNN.output[1]
		end
		xcur = X[X:size(1)]
	end

	local function next_character(pred)
		local probs = softmax:forward(pred)
		if args.temperature then
			local logprobs = torch.log(probs)
			logprobs:div(args.temperature)
			probs = torch.exp(logprobs)
		end
		probs:div(torch.sum(probs))
		local next_char = torch.multinomial(probs:float(), 1):resize(1):float()
		local i = next_char[1]
		local next_text = xrnn.deembed[i] or ''
		return i, next_text
	end

	local n = 1
	repeat
		xrnn.RNN:forward({hcur,xcur})
		hcur = xrnn.RNN.output[1]
		pred = xrnn.RNN.output[2]
		i, next_text = next_character(pred)
		xcur:zero()
		xcur[i] = 1
		if xrnn.pattern == 'word' then
			if not(i==xrnn.numkeys) then
				sample_text = sample_text .. ' ' .. next_text
			end
		else
			sample_text = sample_text .. next_text
		end
		if length then 
			end_condition = (n==length)
			n = n + 1
		else
			end_condition = not(not(next_text:match('\n')))	-- output character includes linebreak
		end
	until end_condition

	if toscreen then
		print('Sample text: ',sample_text)
	end
	if chatmode then
		return sample_text, hcur
	else
		return sample_text
	end
end

function test_conv(temperature)
	local user_input
	local args
	local hcur = torch.Tensor(xrnn.n_hidden):zero()
	repeat
		user_input = io.read()
		io.write('\n')
		io.flush()
		if not(user_input) then
			user_input = ' '
		end
		args = {hcur = hcur, chatmode=true, primetext = user_input .. '\n', temperature = temperature}
		--if length then args.length = length end
		machine_output, hcur = sample_from_network(args)
		io.write('Machine: ' .. machine_output .. '\n')
		io.flush()
	until user_input=="quit"
end

-- SAVING AND LOADING AND CHECKING MEMORY

function memory()
	free,tot = cutorch.getMemoryUsage()
	print(free/tot)
end

function load(filename,gpu)
	-- check some requirements before loading
	local gpu = gpu or -1
	if gpu>=0 then
		require 'cutorch'
		require 'cunn'
	end
	require 'LSTM'
	require 'GRU'
	require 'VanillaRNN'

	T = torch.load(filename)
	xrnn = T.xrnn
	init(xrnn)
end

function loadFromOptions(filename)
	T = torch.load(filename)
	xrnn = T.xrnn
	xrnn.RNN = nil
end

function save(filename)
	torch.save(filename .. '.t7',{xrnn = xrnn})
end
