NOTES 7/15/15:

Hello!

So, uh, this whole thing is still pretty seriously under construction. This is not a polished piece of code yet. This is not ready for consumption. There are some choicy bits but the goals have not yet been realized.

Right now, this does not do seq2seq. This pretty much just exists to learn and generate token sequences. Like Andrej Karpathy's character-RNN, which was a very significant inspiration for this. (In fact, I have flat-out ripped off his model_utils file with no modification.) But, there are some things I've done here which you may find interesting/useful! 

I have implemented LSTM with two potentially desirable features. First, peephole connections, as described here (http://arxiv.org/abs/1503.04069). This lets the gates look at the memory cell contents, instead of just relying on the hidden state. Useful! Second, my LSTM code allows you to make layers of variable sizes. So for example you could have a deep LSTM network where the first layer's hidden state and memory cells are 512-dimensional vectors, and the second layer's are 256-dim, and so on. 

I've also made two layers - joinlayer and splitlayer - that might be useful utility layers. They are probably not maximally memory efficient, but eh, I'll get around to that eventually. The idea of the joinlayer is that it takes a table of vectors in, and concatenates them into one super long vector. And splitlayer takes a super long vector, and splits it into a table of smaller vectors (of sizes you specify in advance). I really should have done the splitting with narrows, but, again, I'll get around to it eventually. 

Also, if you want to split your raw text into word tokens instead of characters, I have made that particularly easy. 

Granted, none of this is really, uh, user friendly yet. If you want to use it, here's how you do:
	+ require 'seq2seq'
	+ build out your list of args, read comments in seq2seq to see what your options are
	+ init(args)
	+ then to train, run 'train_network_N_steps(N)' with some number as the argument
	+ to sample, run 'sample_from_network(length)' with some number as the argument

There's no way to save/load networks yet, or run this from a command line. And again: doesn't do seq2seq yet. But, these things, they are coming. They will be here eventually.
