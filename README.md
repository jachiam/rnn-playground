NOTES 7/25/15:

Hello!

So, uh, this whole thing is still kind of under construction. This is not a well-polished piece of code. But it is probably fun enough to play with that it is worth sharing.

This is a code for building and training RNNs to learn and generate text. Character stuff is easiest, although you can change the matching pattern. (For example, you could learn two-characters at a time, or ignore all punctuation, and so on.) 

This is basically a poor man's version of Andrej Karpathy's char-RNN, which came out earlier this summer, and was awesome. (In fact, I have flat-out ripped off his model_utils file with no modification. The credit and glory here is not all mine.) I spent a good chunk of time learning how to do things in Torch from his and other examples. 

But, there are some things I've done here which you may find interesting/useful! 

I have implemented LSTM with three potentially desirable features. First, peephole connections, as described here (http://arxiv.org/abs/1503.04069). This lets the gates look at the memory cell contents, instead of just relying on the hidden state. Useful! Second, my LSTM code allows you to make layers of variable sizes. So for example you could have a deep LSTM network where the first layer's hidden state and memory cells are 512-dimensional vectors, and the second layer's are 256-dim, and so on. Lastly, forget gate biases are initialized to 1, which is known to help LSTMs perform better at long-term sequence learning tasks. Naive initialization inhibits gradient flow, and this is a quick-fix.

Vanilla RNN and Gated Recurrent Unit implementations are also present and usable. 

I've also made a couple of potentially useful utility layers. Joinlayer and Splitlayer respectively concatenate a table of vectors into one large vector, and split one large vector into a table of smaller vectors. They are probably not maximally memory efficient, but eh, I'll get around to that eventually. I really should have done the splitting with narrows, but, again, eventually. There is also an unbiased linear layer in here (a linear layer with just a weight matrix, and no bias). This is useful if, for instance, you are making gated units, and there should only be one bias per gate instead of two (which is the case in some of the LSTM examples I have seen out there). There's also CosineSimilarity, which I was going to use for something but never got around to it. 

Also, if you want to split your raw text into word tokens instead of characters, I have made that particularly easy. Just use pattern='word' as one of the options. But, uh, to note, that one does not seem to go over super great during training. I'll try adding some better support for it later

So, none of this is really, uh, user friendly yet. If you want to use it, here's how you do:

1. Fire up torch.

2. require 'seq2seq'

3. build out your list of args, read comments in seq2seq to see what your options are

4. init(args)

5. then to train, run 'train_network_N_steps(N)' with some number as the argument

6. or, if you want to proceed through your training corpus all the way some number of times, do 'epochs(N)' where N is reasonable

7. to sample, run 'sample_from_network({length=$some_number})' with some number as the argument

8. if you trained it on a corpus of chat data, try 'test_conv(temperature)' to chat with it!
temperature is a parameter between 0 and 1, which sharpens the probability distribution as you lower it.
that is, it becomes more deterministic for lower temperatures, and more random for higher temps.
I find that temps around 0.65-0.85 are nice.

Saving is easy: just do 'save(filename)'. Loading is also easy: 'load(filename)' does it. But if you want to make a new network with the same options as an old network, do 'loadFromOptions(filename)' and that will do!

I can take requests for additional features. I guess a command line option might be popular? I don't know, I like working with it directly in torch. Get to play with the guts of it while I am going.
