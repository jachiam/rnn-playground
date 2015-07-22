require 'torch'
require 'nn'
require 'nngraph'
require 'joinlayer'
require 'splitlayer'
require 'unbiased_linear'

-- Gated Recurrent Units implementation
function GRU(input_dim,layer_sizes,opt)
	local decoder = opt.decoder or false

	local inputs = {}
	inputs[1] = nn.Identity()()	-- Hidden state input
	inputs[2] = nn.Identity()()	-- External input

	-- How many elements in all the layers?
	local m = 0
	for i=1,#layer_sizes do m = m + layer_sizes[i] end

	-- hidden_split gets a table of the split hidden states
	local hidden_split
	local hidden_states_prev = {}
	if #layer_sizes > 1 then
		hidden_split = nn.SplitLayer(m,layer_sizes)(inputs[1])
		hidden_states_prev = {hidden_split:split(#layer_sizes)}
	else
		hidden_states_prev[1] = inputs[1]
	end

	-- utility function: gives node with linear transform of input
	local function new_input_sum(indim,layer_size,innode,hiddennode)
		local i2h = nn.Linear(indim,layer_size)(innode)
		local h2h = nn.UnbiasedLinear(layer_size,layer_size)(hiddennode)
		return nn.CAddTable()({i2h,h2h})
	end

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

		local z,r,htil

		-- Update gate, z, and reset gate, r
		z = nn.Sigmoid()(new_input_sum(indim,layer_sizes[j],innode,hidden_states_prev[j]))
		r = nn.Sigmoid()(new_input_sum(indim,layer_sizes[j],innode,hidden_states_prev[j]))

		-- 1 - z
		ztil =nn.AddConstant(1,true)(nn.MulConstant(-1,false)(z))

		-- Proposed new memory content
		htil = nn.Tanh()(nn.CAddTable()({
				nn.Linear(indim,layer_sizes[j])(innode),
				nn.CMulTable()({r,nn.Linear(layer_sizes[j],layer_sizes[j])(hidden_states_prev[j])})
				}))

		-- Update
		hidden_states_cur[j] = nn.CAddTable()({
				nn.CMulTable()({hidden_states_prev[j],z}),
				nn.CMulTable()({htil,ztil})
				})

	end

	local external_output_base = hidden_states_cur[#layer_sizes]
	local external_output
	local hidden_state_output 
	if #layer_sizes > 1 then
		hidden_state_output = nn.JoinLayer()(hidden_states_cur)
	else
		hidden_state_output = hidden_states_cur[1]
	end
	if not(decoder) then
		-- I know this looks dumb... but yes, this is intentional.
		-- There's a dumb error with one-layer nets otherwise.
		external_output = nn.Identity()(external_output_base)
	else
		external_output = nn.Linear(layer_sizes[#layer_sizes],input_dim)(external_output_base)
	end
	local outputs = {hidden_state_output,external_output}
	
	return nn.gModule(inputs,outputs)
end
