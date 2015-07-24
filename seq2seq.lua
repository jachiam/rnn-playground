require 'torch'
require 'nn'
require 'nngraph'
local model_utils = require 'model_utils'
package.path = package.path .. ";./network/?.lua"

s2s = {}

--[[ Arguments:
	gpu
	batch_size
	seq_len
	collect_often
	noclones	-- if true, gradients are calculated more slowly but much more memory-efficiently (default false)
	report_freq
	conv_verbose	-- if true, when conversing, if input symbol is unknown, declare so

	RNN_type	-- string, determines RNN type. options are 'LSTM', 'GRU', 'Vanilla'.

	training_mode	-- if 1, randomly samples minibatches from anywhere in text corpus (default)
			-- if 2, proceeds linearly through text corpus

	lr
	lambda
	gamma
	grad_clip
	clip_to

	pattern
	lower
	rawdata
	usemostcommon	-- use only the most common pattern-matching entries in rawdata as input symbols
	useNmostcommon  -- how many?

	layer_sizes
	peepholes	-- for LSTM networks
	nl_type		-- for Vanilla RNN networks
]]
function init(args)

	-- Verbosity
	s2s.report_freq = args.report_freq or 10
	s2s.conv_verbose = false

	-- Computational
	s2s.gpu = args.gpu or -1
	if s2s.gpu >= 0 then
		require 'cutorch'
		require 'cunn'
	end
	s2s.batch_size = args.batch_size or 30
	s2s.seq_len = args.seq_len or 50
	s2s.collect_often = args.collect_often or false
	s2s.noclones = args.noclones or false

	-- RMSprop and clipping gradients
	s2s.lr = args.lr or 0.01		-- learning rate
	s2s.lambda = args.lambda or 1e-8
	s2s.gamma = args.gamma or 0.95
	s2s.grad_clip = args.grad_clip or true
	s2s.clip_to = args.clip_to or 5

	-- Training
	s2s.training_mode = args.training_mode or 1

	-- Data
	s2s.pattern = args.pattern or '.'
	s2s.lower = args.lower or false
	s2s.usemostcommon = args.usemostcommon or false
	s2s.useNmostcommon = args.useNmostcommon or 4500
	s2s.replace = args.replace or false
	s2s.wordopt = {usemostcommon = s2s.usemostcommon, useNmostcommon = s2s.useNmostcommon, replace = s2s.replace}
	s2s.filename = args.filename or 'input.txt'
	s2s.rawdata = args.rawdata or data(s2s.filename)
	print('Creating embedding/deembedding tables for characters in data...')
	s2s.embed, s2s.deembed, s2s.numkeys, s2s.numwords, s2s.tokenized, s2s.freq_data = data_processing(s2s.rawdata,s2s.pattern,s2s.lower,s2s.wordopt)
	s2s.numkeys = s2s.numkeys + 1 		-- for unknown character
	s2s.eye = torch.eye(s2s.numkeys)
	print('Finished making embed/deembed tables.') 
	print('Finished making embedded data.')
	print('Dictionary has this many words in it: ',s2s.numkeys)

	-- Networks and parameters
	s2s.layer_sizes = args.layer_sizes or {128}
	s2s.peepholes = args.peepholes or false
	s2s.nl_type = args.nl_type or 'tanh'
	s2s.n_hidden = 0

	s2s.RNN_type = args.RNN_type or 'LSTM'

	if s2s.RNN_type == 'LSTM' then
		print('Making LSTM...')
		for i=1,#s2s.layer_sizes do s2s.n_hidden = s2s.n_hidden + 2*s2s.layer_sizes[i] end
		require 'LSTM'
	elseif s2s.RNN_type == 'GRU' then
		print('Making GRU...')
		for i=1,#s2s.layer_sizes do s2s.n_hidden = s2s.n_hidden + s2s.layer_sizes[i] end
		require 'GRU'
	elseif s2s.RNN_type == 'Vanilla' then
		print('Making Vanilla RNN...')
		for i=1,#s2s.layer_sizes do s2s.n_hidden = s2s.n_hidden + s2s.layer_sizes[i] end
		require 'VanillaRNN'
	end

	clones = {}
	if args.RNN then
		s2s.RNN = args.RNN
	elseif s2s.RNN_type == 'LSTM' then
		s2s.RNN = LSTM(s2s.numkeys,s2s.layer_sizes,{peepholes=s2s.peepholes,decoder=true,fgate_init=true})
	elseif s2s.RNN_type == 'GRU' then
		s2s.RNN = GRU(s2s.numkeys,s2s.layer_sizes,{decoder=true})
	elseif s2s.RNN_type == 'Vanilla' then
		s2s.RNN = VanillaRNN(s2s.numkeys,s2s.layer_sizes,{decoder=true,nl_type=s2s.nl_type})
	end
	if s2s.gpu >= 0 then s2s.RNN:cuda() end
	s2s.params, s2s.gradparams = s2s.RNN:getParameters()
	s2s.v = s2s.gradparams:clone():zero()
	if not(s2s.noclones) then
		print('Cloning RNN...')
		clones.RNN = model_utils.clone_many_times(s2s.RNN,s2s.seq_len)
		collectgarbage()
	end
	print('RNN is done.')

	print('Making criterion...')
	local criterion_input_1, criterion_input_2, criterion_out
	criterion_input_1 = nn.Identity()()
	criterion_input_2 = nn.Identity()()
	criterion_out = nn.ClassNLLCriterion()({nn.LogSoftMax()(criterion_input_1),criterion_input_2})
	s2s.criterion = nn.gModule({criterion_input_1,criterion_input_2},{criterion_out})
	if s2s.gpu >= 0 then s2s.criterion:cuda() end
	if not(s2s.noclones) then
		print('Cloning criterion...')
		clones.criterion = model_utils.clone_many_times(s2s.criterion,s2s.seq_len)
		collectgarbage()
	end
	print('Criterion is done.')
	
	if s2s.gpu >=0 then
		free,tot = cutorch.getMemoryUsage()
		print('Free fraction of memory remaining: ', free/tot)
	end
	print('Number of trainable parameters: ', s2s.params:nElement())

end


-- DATA MANIPULATION FUNCTIONS
function data(filename)
	local f = torch.DiskFile(filename)
	local rawdata = f:readString('*a')
	f:close()
	return rawdata
end

function split_string(rawdata,pattern,lower,wordopt)
	local replace 
	if wordopt then
		replace = wordopt.replace or false
	end
	if pattern == 'word' then
		--ptrn = '%w+\'?%w*[%s%p]'
		ptrn = '[^%s]+\n?'
	else
		ptrn = pattern
	end
	local breakapart = rawdata:gmatch(ptrn)
	local splitstring = {}
	local tokens = {}
	for elem in breakapart do
		tokens = {}
		if lower then elem = elem:lower() end
		if pattern == 'word' then
			local pref = {}
			local front = elem
			local back = {}

			--[[ strip off newline characters
			back[1] = elem:sub(elem:len(),elem:len())
			if back[1] == '\n' then 
				front = elem:sub(1,elem:len()-1)
			else
				front = elem
			end]]

			-- strip of punctuation characters and newlines
			for i=1,front:len() do
				local prevchar = front:sub(1,1)
				if prevchar:match('[%p\n]') then
					table.insert(pref,prevchar)
					front = front:sub(2,front:len())
				else
					break
				end
			end
			for i=front:len(),1,-1 do
				local lastchar = front:sub(front:len(),front:len())
				if lastchar:match('[%p\n]') then
					table.insert(back,lastchar)
					front = front:sub(1,front:len()-1)
				else
					break
				end
			end

			-- prefix characters/punctuation to tokens
			for i=1,#pref do
				tokens[#tokens+1] = pref[i]
			end

			-- word to token
			-- time for some common replacements!
			if replace and front then
				local asplit = {}
				local ba = front:gmatch('[^\']+')
				for a in ba do
					table.insert(asplit,a)
				end
				local replaceflag = false
				if #asplit > 1 then
					local prev = asplit[#asplit-1]:lower()
					local last = asplit[#asplit]:lower()
					if last == 'll' then
						asplit[#asplit] = 'will'
						replaceflag = true
					elseif last == 'm' then
						asplit[#asplit] = 'am'
						replaceflag = true
					elseif last == 've' then
						asplit[#asplit] = 'have'
						replaceflag = true
					elseif last == 're' then
						asplit[#asplit] = 'are'
						replaceflag = true
					elseif last == 's' then
						if prev == 'he' or prev == 'she' 
							or prev == 'that' or prev == 'this'
							or prev == 'it' or prev == 'how' 
							or prev == 'why' or prev == 'who'
							or prev == 'when' or prev == 'what' then
							asplit[#asplit] = 'is'
							replaceflag = true
						end
					end
				end
				if not(replaceflag) then
					tokens[#tokens+1] = front
				else
					for i=1,#asplit do
						tokens[#tokens+1] = asplit[i]
					end
				end
			else
				tokens[1] = front
			end

			--suffic characters/punctuation to tokens
			for i=#back,1,-1 do
				tokens[#tokens+1] = back[i]
			end
		else
			tokens[1] = elem
		end
		for i,v in pairs(tokens) do table.insert(splitstring,tokens[i]) end
	end
	return splitstring
end

function data_processing(rawdata,pattern,lower,wordopt)
	local usemostcommon = false
	local useNmostcommon = 4500
	if wordopt then
		usemostcommon = wordopt.usemostcommon or false
		useNmostcommon = wordopt.useNmostcommon or 4500
	end
	local embeddings = {}
	local deembeddings = {}
	local freq = {}
	local numkeys = 0
	local numwords = 0

	-- split the string and make embeddings/deembeddings/freq
	local splitstring = split_string(rawdata,pattern,lower,wordopt)
	numwords = #splitstring
	tokenized = torch.zeros(numwords)
	for i=1,numwords do
		if not embeddings[splitstring[i]] then
			numkeys = numkeys + 1
			embeddings[splitstring[i]] = numkeys
			deembeddings[numkeys] = splitstring[i]
			freq[numkeys] = {1,numkeys}
		else
			freq[embeddings[splitstring[i]]][1] = freq[embeddings[splitstring[i]]][1] + 1
		end
		tokenized[i] = embeddings[splitstring[i]]
	end

	-- only take the most frequent entries
	local num_represented = 0
	if usemostcommon then
		numkeys = math.min(numkeys,useNmostcommon)
		table.sort(freq,function(a,b) return a[1]>b[1] end)
		local new_embed = {}
		local new_deembed = {}
		for i=1,numkeys do
			new_deembed[i] = deembeddings[freq[i][2]]
			new_embed[new_deembed[i]] = i
			num_represented = num_represented + freq[i][1]
		end
		embeddings = new_embed
		deembeddings = new_deembed
		print('Dictionary captures about ', 100*num_represented/numwords, '% of text.')
		-- rebuild tokenized:
		for i=1,numwords do
			tokenized[i] = embeddings[splitstring[i]] or numkeys + 1
		end
	end

	return embeddings, deembeddings, numkeys, numwords, tokenized, freq
end


-- TRAINING

function fwd()
	total_loss = 0
	for i=1,s2s.seq_len do
		clones.RNN[i]:forward({H[i],X[i]})
		if i < s2s.seq_len then H[i+1] = clones.RNN[i].output[1] end
		loss = clones.criterion[i]:forward({clones.RNN[i].output[2],Y[i]})
		total_loss = total_loss + loss[1]
	end
	if s2s.collect_often then collectgarbage() end
end

function bwd()
	for i=s2s.seq_len,1,-1 do
		clones.criterion[i]:backward({clones.RNN[i].output[2],Y[i]},{1})
		if i < s2s.seq_len then
			clones.RNN[i]:backward({H[i],X[i]},{
				clones.RNN[i+1].gradInput[1],
				clones.criterion[i].gradInput[1]
				})
		else
			clones.RNN[i]:backward({H[i],X[i]},{
				torch.Tensor():typeAs(H[i]):resizeAs(H[i]):zero(),
				clones.criterion[i].gradInput[1]
				})
		end
		if s2s.collect_often then collectgarbage() end
	end
end

function grad_pass_no_clones()
	total_loss = 0

	outputs = {}
	-- fwd pass to get the outputs and hidden states
	for i=1,s2s.seq_len do
		s2s.RNN:forward({H[i],X[i]})
		if i < s2s.seq_len then H[i+1] = s2s.RNN.output[1] end
		outputs[i] = s2s.RNN.output[2]
	end
	if s2s.collect_often then collectgarbage() end

	gradInputs = {}
	-- bwd pass
	for i=s2s.seq_len,1,-1 do
		loss = s2s.criterion:forward({outputs[i],Y[i]})
		total_loss = total_loss + loss[1]
		s2s.criterion:backward({outputs[i],Y[i]},{1})
		s2s.RNN:forward({H[i],X[i]})
		if i < s2s.seq_len then
			s2s.RNN:backward({H[i],X[i]},{gradInputs[i], s2s.criterion.gradInput[1]})
		else
			s2s.RNN:backward({H[i],X[i]},{torch.Tensor():typeAs(H[i]):resizeAs(H[i]):zero(),s2s.criterion.gradInput[1]})
		end
		gradInputs[i-1] = s2s.RNN.gradInput[1]
	end
	if s2s.collect_often then collectgarbage() end
end

-- First index is time slice
-- Second is element-in-batch
function minibatch_loader()
	local i
	local preX
	local X = torch.Tensor(s2s.seq_len,s2s.batch_size,s2s.numkeys):zero()
	local Y = torch.Tensor(s2s.seq_len,s2s.batch_size):zero()
	local H = torch.Tensor(s2s.seq_len,s2s.batch_size,s2s.n_hidden):zero()

	if s2s.training_mode == 2 then
		if not(s2s.pos_in_text) then
			s2s.pos_in_text = 1
		end
	end

	for n=1,s2s.batch_size do
		if s2s.training_mode == 1 then
			i = torch.ceil(torch.uniform()*(s2s.numwords - s2s.seq_len))
		else
			i = s2s.pos_in_text
		end
		preX = s2s.tokenized[{{i,i + s2s.seq_len - 1}}]:long()
		X[{{},{n}}]:copy(s2s.eye:index(1,preX))
		Y[{{},{n}}] = s2s.tokenized[{{i+1,i + s2s.seq_len}}]
		if s2s.training_mode == 2 then
			s2s.pos_in_text = s2s.pos_in_text + s2s.seq_len
			if s2s.pos_in_text > s2s.numwords - s2s.seq_len then
				s2s.pos_in_text = 1
			end
		end
	end

	if s2s.gpu >= 0 then
		X = X:float():cuda()
		Y = Y:float():cuda()
		H = H:float():cuda()
	end

	return X,Y,H
end


function train_network_one_step()
	X,Y,H = minibatch_loader()
	s2s.gradparams:zero()

	if s2s.noclones then
		grad_pass_no_clones()
	else
		fwd()
		bwd()
	end

	-- Average over batch and sequence length
	s2s.gradparams:div(s2s.batch_size):div(s2s.seq_len)

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
		if n==1 then init_error = total_loss/s2s.seq_len end
		if total_loss/s2s.seq_len > 3*init_error then
			print('Error is exploding. Current error: ', total_loss/s2s.seq_len)
			print('Terminating training here.')
			break
		end
		running_total_loss = running_total_loss + total_loss/s2s.seq_len
		if n % s2s.report_freq == 0 then 
			if s2s.training_mode == 1 then
				print('Average Error: ',running_total_loss/s2s.report_freq,'Num Steps: ',n)
			else
				print('Average Error: ',running_total_loss/s2s.report_freq,'Num Steps: ',n,' % thru text: ', round(100*s2s.pos_in_text/s2s.numwords))
			end 
			running_total_loss = 0
		end
	end
end

function round(x)
	return math.floor(x * 1000)/1000
end

-- SAMPLING

function tokenize_string(text)

	local splitstring = split_string(text,s2s.pattern,s2s.lower,s2s.wordopt)
	local numtokens = #splitstring
	local tokenized = torch.zeros(numtokens)
	for i=1,numtokens do
		tokenized[i] = s2s.embed[splitstring[i]] or s2s.numkeys
		if tokenized[i] == s2s.numkeys and s2s.conv_verbose then
			print('Machine Subconscious: I did not recognize the word ' .. splitstring[i] .. '.')
			print()
		end
	end

	return tokenized:long(), numtokens
end

function string_to_one_hot(text)
	local tokenized, numtokens = tokenize_string(text)
	local one_hots = torch.zeros(numtokens,s2s.numkeys)
	one_hots:copy(s2s.eye:index(1,tokenized))
	return one_hots
end

function sample_from_network(args)
	local X, xcur, hcur
	local length = args.length
	local toscreen = args.toscreen or false
	local primetext = args.primetext
	sample_text = ''

	xcur = torch.Tensor(s2s.numkeys):zero()
	hcur = args.hcur or torch.Tensor(s2s.n_hidden):zero()
	softmax = nn.SoftMax()
	if s2s.gpu >= 1 then
		xcur = xcur:float():cuda()
		hcur = hcur:float():cuda()
		softmax:cuda()
	end

	if not(primetext) then
		i = torch.ceil(torch.uniform()*s2s.numkeys)
		xcur[i] = 1
	else
		X = string_to_one_hot(primetext)
		if s2s.gpu >=1 then
			X = X:cuda()
		end
		for n=1,X:size(1)-1 do
			s2s.RNN:forward({hcur,X[n]})
			hcur = s2s.RNN.output[1]
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
		local next_text = s2s.deembed[i] or ''
		return i, next_text
	end

	if length then
		for n=1,length do
			s2s.RNN:forward({hcur,xcur})
			hcur = s2s.RNN.output[1]
			pred = s2s.RNN.output[2]
			i, next_text = next_character(pred)
			xcur:zero()
			xcur[i] = 1
			if s2s.pattern == 'word' then
				if not(i==s2s.numkeys) then
					sample_text = sample_text .. ' ' .. next_text
				end
			else
				sample_text = sample_text .. next_text
			end
		end
	else
		repeat
			s2s.RNN:forward({hcur,xcur})
			hcur = s2s.RNN.output[1]
			pred = s2s.RNN.output[2]
			i, next_text = next_character(pred)
			xcur:zero()
			xcur[i] = 1
			if s2s.pattern == 'word' then
				if not(i==s2s.numkeys) then
					sample_text = sample_text .. ' ' .. next_text
				end
			else
				sample_text = sample_text .. next_text
			end
		until next_text == '\n'	-- output character is unknown/EOS
	end

	if toscreen then
		print('Sample text: ',sample_text)
	end
	return sample_text, hcur
end

function test_conv(temperature)
	local user_input
	local args
	local hcur = torch.Tensor(s2s.n_hidden):zero()
	repeat
		user_input = io.read()
		io.write('\n')
		io.flush()
		if not(user_input) then
			user_input = ' '
		end
		args = {hcur = hcur, toscreen=false, primetext = user_input .. '\n', temperature = temperature}
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
	s2s = T.s2s
	init(s2s)
end

function loadFromOptions(filename)
	T = torch.load(filename)
	s2s = T.s2s
	s2s.RNN = nil
end

function save(filename)
	torch.save(filename .. '.t7',{s2s = s2s})
end
