---
layout: post
title: Attention
abstract: A short text explaining what this post is about.
tags: [tag1, tag2, tag3]
category: category
mathjax: false
update: 0000-01-01 
time: 0
words: 0
---

# {{ page.title }}

- Sequence-to-sequence (seq2seq): Input is a sequence of vectors (or tensors), output is also a sequence of vectors (or tensors) of the same length
- Model can adapt to varying sequence length
- Autoregressive training: Unsupervised training where the target is to predict the next element in a sequence (target sequence is input sequence shifted one (time) step to the left)
- Models conditional probability distribution of next token given all previous tokens
- Seq2seq can also do label-to-sequence and sequence-to-label (label = sequence of length one)
- Causal vs. non-causal: Model can only look at previous sequence elements (next-word-prediction) or previous and next elements (language modeling)
- Embedding = input in feature space (convolutional feature map, autoencoder latent space), word2vec (NLP)
- Distributional hypothesis: "You shall know a word by the company it keeps."
- Conv, RNN and Self-Attention are seq2seq, MLP only when shared
- One-hot representation is sparse (waste of space)
- Embedding representation is dense and needs less elements to represent the input because of large number of possible combinations
- One-hot vector times matrix = embedding (selects on row/column of the matrix)
- Seq2lable: global pooling of output sequence or "global unit" (last unit is supposed to predict the label)
- Label2seq: Image captioning
- Label+seq2seq: (1) Image captioning with autoregressive caption prediction, (2) Teacher forcing: Encoder-decoder in machine translation: Encoder encodes sentence in input language into "label" (embedding/feature representation), decoder decodes encoder encoding _and_ autoregressive next word prediction in output language
- Encoder is non-causal, decoder is causal
- Backprop through time: Unrolled RNN
- RNN downsides: Slow (sequential computation), vanishing/exploding gradients
- RNN upsides: Potentially unbounded memory (receptive field over entire sequence contrary to CNNs)
- RNN label2seq: Label is initial hidden state
- RNN seq2label: Last hidden state is label
- Conv1D can be used for time series (sequences)
- Made causal by padding asymmetrically
- Needs dilated filters to deal with long range, make dense by stacking convs with different dilation factors (dilation factors are hyperparameters)
- Backprop through number of layers instead number of time steps; Allows parallel training instead of sequential
- Word2Vec: Learn to predict words around words in sliding window fashion using a single bottleneck hidden layer linear NN (i.e. linera one-layer autoencoder)
- Latent AE factors are word embeddings
- Self-attention advantages: Parallel computation and long-range dependencies
- Basic self-attention: Output is weighted sum over the input: $y_i=\sum_j w_{ij}x_j$
- Difference to FC feed-forward NN: $w_{ij}$ is not a learnable parameter but an attention score between two input tokens ($w_{ij}=softmax(x_i^Tx_j)$)
- Dot-product self-atttention: softmax(key.query)*value
- Vectorize: $W=softmax(X^TX)$ (row-wise softmax); $Y^T=WX^T$
- Diagonal of W is largest in this basic setup; no parameters
- Linear computation between X and Y (strong gradients)
- Same distance from every input to every output (pair-wise self-attention): Different form RNNs where distance grows
- Self-attention is a set model not a sequence model: Sequential information needs to be added on top (positional embedding)
- Permutation equivariant: p(sa(X)) = sa(p(X))
- Power of dot-product: Respects magnitude and sign (recommender systems)
- Scaled dot-product by sqrt(k) where k is the number of elements: Prevents vanishing gradient in softmax
- Attention as "soft" dict (key, value): Every key matches the query to some extent and a mixture of values is returned
- Liner transformations on X: K, Q, V allow the key, query and value to play different roles
- Multi-head self-attention (multiple self-attention operations applied in parallel): A token can relate to another in more than one way
- Down-project input with one linear transformation per head, concatenate outputs, apply a single linear transformation
- Same number of parameters: Linear transforms/output is simply cut into head-sized pieces

## Code & References

| [Code](/url/to/notebook.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](/url/to/binder/notebook.ipynb)                                                         |   |
|:------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------|:-:|
|                                |                                                                                                                                         |   |
|              [1]               | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer)                                                       |   |
|              [2]               | [The Narrated Transformer](https://www.youtube.com/watch?v=-QH8fRhqFHM)                                                                 |   |
|              [3]               | [Transformer Neural Networks](https://www.youtube.com/watch?v=TQQlZhbC5ps)                                                              |   |
|              [4]               | [Bert Neural Network](https://www.youtube.com/watch?v=xI0HHN5XKDo)                                                                      |   |
|              [5]               | [Attention in Neural Networks](https://www.youtube.com/watch?v=W2rWgXJBZhU)                                                             |   |
|              [6]               | [Self-Attention](https://www.youtube.com/watch?v=KmAISyVvE1Y)                                                                           | x |
|              [7]               | [Transformers](https://www.youtube.com/watch?v=oUhGZMCTHtI)                                                                             |   |
|              [8]               | [Famous Transformers](https://www.youtube.com/watch?v=MN__lSncZBs)                                                                      |   |
|              [9]               | [Transformers from Scratch](http://peterbloem.nl/blog/transformers)                                                                     |   |
|              [10]              | [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)                                                      |   |
|              [11]              | [Visualizing a NMT Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention) |   |
|              [12]              | [Transformer (Google AI Blog)](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)                                 |   |
|              [13]              | [Attention is all you need (video)](https://www.youtube.com/watch?v=rBCqOTEfxvg)                                                        |   |
|              [14]              | [Attention Is All You Need (paper)](https://arxiv.org/abs/1706.03762)                                                                   |   |
|              [15]              | [What are Transformers?](https://www.youtube.com/watch?v=XSSTuhyAmnI)                                                                   |   |
|              [16]              | [Illustrated Guide to Transformers](https://www.youtube.com/watch?v=4Bdc55j80l8)                                                        |   |
|              [17]              | [An Introduction to Attention](https://wandb.ai/authors/under-attention/reports/An-Introduction-to-Attention--Vmlldzo1MzQwMTU)          |   |
