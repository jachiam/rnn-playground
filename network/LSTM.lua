require 'torch'
require 'nn'
require 'nngraph'
require 'joinlayer'
require 'splitlayer'
require 'unbiased_linear'

-- layer_sizes is a table whose entries are the number of cells per layer
-- fgate_init is a flag: if true, then initialize the forget gate biases to 1
function LSTM(input_dim,layer_sizes,opt)
	local peep = opt.peepholes or false
	local decoder = opt.decoder or false
	local fgate_init = opt.fgate_init or false

	local inputs = {}
	inputs[1] = nn.Identity()()	-- Hidden state input
	inputs[2] = nn.Identity()()	-- External input

	-- How many elements in all the layers?
	local m = 0
	for i=1,#layer_sizes do m = m + layer_sizes[i] end

	local sizes_with_cells = {}
	for i=1,#layer_sizes do
		sizes_with_cells[i] = layer_sizes[i]
		sizes_with_cells[i + #layer_sizes] = layer_sizes[i]
	end

	-- hidden_split gets a table of the split hidden states
	local hidden_split = nn.SplitLayer(2*m,sizes_with_cells)(inputs[1])
	local hidden_states_prev = {hidden_split:split(2*#layer_sizes)}

	-- utility function: gives node with linear transform of input
	local function new_input_sum(indim,layer_size,innode,hiddennode,biasflag)
		local biasflag = biasflag or false
		local i2h = nn.Linear(indim,layer_size)(innode)
		local h2h = nn.UnbiasedLinear(layer_size,layer_size)(hiddennode)
		if biasflag then
			i2h.data.module.bias:fill(1)
		end
		return nn.CAddTable()({i2h,h2h})
	end

	-- we will assume the following structure in the hidden state:
	-- the first /k/ entries are the h_j, and the second /k/ entries
	-- are the cell memory states c_j. 
	local hidden_states_cur = {}
	for j=1,#layer_sizes do
		local innode, indim
		if j==1 then
			innode = inputs[2]
			indim = input_dim
		else
			innode = hidden_states_cur[j-1]
			indim = layer_sizes[j-1]
		end

		local zbar, ibar, fbar, obar, z, i, f, o, p_i, p_f, p_o

		-- Input block, input gate, forget gate, output gate linear transforms.
		zbar = new_input_sum(indim,layer_sizes[j],innode,hidden_states_prev[j])
		ibar = new_input_sum(indim,layer_sizes[j],innode,hidden_states_prev[j])
		fbar = new_input_sum(indim,layer_sizes[j],innode,hidden_states_prev[j],fgate_init)
		obar = new_input_sum(indim,layer_sizes[j],innode,hidden_states_prev[j])

		-- Input block nonlinear
		z = nn.Tanh()(zbar)

		-- Input and forget gate nonlinear / and possibly peepholes
		if not(peep) then
			i = nn.Sigmoid()(ibar)
			f = nn.Sigmoid()(fbar)
		else
			p_i = nn.CMul(layer_sizes[j])(hidden_states_prev[j+#layer_sizes])
			p_f = nn.CMul(layer_sizes[j])(hidden_states_prev[j+#layer_sizes])
			i = nn.Sigmoid()(nn.CAddTable()({ibar,p_f}))
			f = nn.Sigmoid()(nn.CAddTable()({fbar,p_f}))
		end

		-- Calculate memory cell values
		-- hidden_states_cur[j + #layer_sizes] is c_t for this layer
		hidden_states_cur[j + #layer_sizes] = nn.CAddTable()({
			nn.CMulTable()({z,i}), 
			nn.CMulTable()({f,hidden_states_prev[j + #layer_sizes]})
			})

		-- Output Gate nonlinear / and possibly peepholes
		if not(peep) then
			o = nn.Sigmoid()(obar)
		else
			p_o = nn.CMul(layer_sizes[j])(hidden_states_cur[j+#layer_sizes])
			o = nn.Sigmoid()(nn.CAddTable()({obar,p_o}))
		end

		-- now make h_t for this layer
		hidden_states_cur[j] = nn.CMulTable()({
			nn.Tanh()(hidden_states_cur[j + #layer_sizes]), o
			})
	end

	local external_output_base = hidden_states_cur[#layer_sizes]
	local external_output
	local hidden_state_output = nn.JoinLayer()(hidden_states_cur)
	if not(decoder) then
		external_output = external_output_base
	else
		external_output = nn.Linear(layer_sizes[#layer_sizes],input_dim)(external_output_base)
	end
	local outputs = {hidden_state_output,external_output}
	
	return nn.gModule(inputs,outputs)
end
