require 'torch'
require 'nn'
require 'nngraph'
require 'joinlayer'
require 'splitlayer'
require 'peephole_layer'

-- layer_sizes is a table whose entries are the number of cells per layer

function LSTM(input_dim,layer_sizes, peepholes, decoder)
	local peep = peepholes or false
	local decoder = decoder or false

	-- For simplicity, there are two inputs. The hidden state (which includes
	-- the hidden state at /every/ level, and the memory cells), and the 
	-- external input.

	-- Okay, so, 'simplicity' may be a lie. It makes things quite hairy in here
	-- at the start of things, because we have to unpack the hidden state, and
	-- then repack it later. But it'll make this easy to plug-n-play in a simple
	-- trainer I've built for another piece of code, which is the point of the 
	-- exercise. And you should have a very easy time writing your own code to
	-- train it. 

	-- There'll be two outputs: first, an "external output," which will be the 
	-- last layer's hidden state (but not its memory cell). This is ostensibly
	-- what we will send as input to the next network. The second output will be
	-- the complete updated hidden state (including the hidden state at every
	-- level, plus memory cells) which you can then feet back into the LSTM as
	-- input.

	-- If the 'decoder' flag is up, then that means we expect to use the output
	-- for classification purposes, so we have to put a final-layer decoder in here.

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
		-- Following the example from Karpathy's code...
		-- Do all of the linear transforms from input and recurrents in one go.
		-- The factor of 4 is because there are four linear transformations required:
		-- One for the input block, z; one for the input gate, i; one for the
		-- forget gate, f; and one for the output gate, o.
		local i2h = nn.Linear(indim,4*layer_sizes[j])(innode)
		local h2h = nn.Linear(layer_sizes[j],4*layer_sizes[j])(hidden_states_prev[j])
		local all = nn.CAddTable()({i2h,h2h})

		local sizes = {layer_sizes[j], layer_sizes[j], layer_sizes[j], layer_sizes[j]}
		local zbar, ibar, fbar, obar, z, i, f, o, p_i, p_f, p_o
		zbar, ibar, fbar, obar = nn.SplitLayer(4*layer_sizes[j],sizes)(all):split(4)

		-- Input Block
		z = nn.Tanh()(zbar)

		-- Input and Forget Gates
		if not(peep) then
			i = nn.Sigmoid()(ibar)
			f = nn.Sigmoid()(fbar)
		else
			p_i = nn.PeepLayer(layer_sizes[j])(hidden_states_prev[j+#layer_sizes])
			p_f = nn.PeepLayer(layer_sizes[j])(hidden_states_prev[j+#layer_sizes])
			i = nn.Sigmoid()(nn.CAddTable()({ibar,p_f}))
			f = nn.Sigmoid()(nn.CAddTable()({fbar,p_f}))
		end

		-- hidden_states_cur[j + #layer_sizes] is c_t for this layer
		hidden_states_cur[j + #layer_sizes] = nn.CAddTable()({
			nn.CMulTable()({z,i}), 
			nn.CMulTable()({f,hidden_states_prev[j + #layer_sizes]})
			})

		-- Output Gate
		if not(peep) then
			o = nn.Sigmoid()(obar)
		else
			p_o = nn.PeepLayer(layer_sizes[j])(hidden_states_cur[j+#layer_sizes])
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
