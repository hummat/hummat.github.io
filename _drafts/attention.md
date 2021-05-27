---
layout: post
title: Attention
abstract: Yet another article on attention? Yes, but this one is both annotated and illustrated focusing on attention itself instead of the architecture making it famous. After all, attention is all you need, not Transformers.
tags: [attention, context]
category: learnig
thumbnail: /images/attention/query_key_value.png
mathjax: true
jquery: true
time: 0
words: 0
---

# {{ page.title }}

You might be wondering if this article can possible contain anything new for you. You already studied the [illustrated]() and [annotated]() Transformer, the [original paper]() and everything from [GPT-1-2-3]() to [BERT]() and beyond. Well, if you have, it could indeed be that there is nothing _fundamentally_ new for you to be found here, but the goal to weave all this information into one coherent story and provide further context across dimensions and domains with a focus on attention itself instead of the scaffolding erected around it referred to as _The Transformer_ has potential to further clarify and solidify some of these truly interesting and general concepts.

### Notation

Our detailed explanation of the attention mechanism begins with some math which will subsequently be visualized to build an intuitive understanding. A quick word on notation: The attention equation makes use of scalars, vectors and matrices. We will use lowercase letters $x$, bold lowercase letters $\boldsymbol{x}$ and uppercase letters $W$ for these entities respectively. A great way to visualize scalars, vectors and matrices is to represent them by colored squares, where the number of squares represents the dimensionality and the color vibrancy of each square represents the magnitude of the value at this position from small, bright values to large, dark values as shown below.

<img class="img-animate" src="/images/attention/notation.png">

### The basic attention equation

Making use of this notation, the basic attention equation can be written as

$$
\boldsymbol{y}_j = \sum_{i=1}^Nw_{ij}\boldsymbol{x}_i
$$

So what's happening here? In a nutshell: The output of the attention operation $\boldsymbol{y}\_j$ is a weighted sum of the inputs $\boldsymbol{x}\_i$. The inputs are the set elements introduced in the [previous article](), for example words, pixels or points in 3D space. They are usually vectors, where the dimensions, commonly called input features, can represent various properties like RGB color channels or XYZ Euclidean coordinates. Words are somewhat special in this regard, as they don't have any intrinsic features and are therefore commonly represented by a large vector with values encoding their semantics computed from their co-occurrence with other words called an _embedding_[^1].

[^1]: Search for _"word2vec"_ if you want further details. I recommend [this blog post](https://jalammar.github.io/illustrated-word2vec).

### The dot product and softmax

What are those weights $w\_{ij}$ then and how are they determined? Though we employ attention in the deep learning domain, in it's most basic form, it actually doesn't feature any learnable parameters. Instead, the weights are computed from the inputs using the dot product as similarity measure: $w\_{ij}=\boldsymbol{x}\_i^T\boldsymbol{x}\_j$. To understand how this works and why it makes sense, let's take a look at the visualization below.

<img class="img-animate" src="/images/attention/dot_product.png">

The two vectors $\boldsymbol{u}$ and $\boldsymbol{m}$ represent the movie preferences of a user and the movie content respectively with three features each. Now, taking the dot product, we get a score representing the match between user and movie. Note that this takes into account both the magnitude of the values, as multiplying large values results in a large increase of the score, as well as the sign. Imagine a scale between $-1$ and $1$ where the former represent strong dislike (or weak occurance in case of the movie vector) and the latter a strong inclination (or strong occurance). Now, given a user who dislikes action and a movie with little action, the score will still be high as both negative signs cancel out, which is exactly what we want.
To make those weights more interpretable and prevent the output from becoming excessively large, we can normalize all weights $\boldsymbol{w}\_j=\{w\_{ij}\}\_{i=1}^N$ to fall between $0$ and $1$ using the _softmax_ function which is reminiscent of its application on the logits in classification tasks. We can think about the resulting normalized weight vector as a categorical probability distribution over the inputs where high probability is assigned to inputs with strong similarity[^2].

### Attention visualized

Once all weights are computed, we multiply them with their corresponding inputs and sum them up to obtain the $j^{th}$ output which, as stated before, is simply a weighted sum of the inputs[^3]. The computation graph below visualizes the entire process.

[^2]: Keep in mind though, that this is not a real probability distribution where the probability values correspond to empirical frequencies but rather a miscalibrated approximation.
[^3]: This is omitted in the basic attention equation but we will include it in the upcoming matrix notation.

<img class="img-animate" src="/images/attention/attention.png">

As in the dot product example, think of the transposed vectors (the horizontal ones) as individual users and the vertical ones as movies. The outputs would then represent something akin to an ideal movie, one for each user, stitched together from the three individual movies on offer.

### Queries, keys and values

But wait, we aren't actually dealing with different entities like users and movies in this basic formulation of attention, as all three vectors involved---two of which are used to compute the weight and the remaining one as part of the weighted sum---stem from the same set $X$. This also means that output $\boldsymbol{y}\_1$ will be dominated by input $\boldsymbol{x}\_1$ as the dot product between two identical vectors is usually large (after all, they are as similar as it gets). To deal with these issues, we can try to make the attention computation more versatile and expressive through the introduction of three learnable parameter matrices $W^Q$, $W^K$ and $W^V$. Through multiplication with the inputs $\boldsymbol{x}$ we obtain three distinct vectors for the three roles on offer namely the <span style="color: #AB63FA; font-weight: bold;">query $\boldsymbol{q}$</span>, <span style="color: #FECB52; font-weight: bold;">key $\boldsymbol{k}$</span> and <span style="color: #00CC96; font-weight: bold;">value $\boldsymbol{v}$</span>.

$$
\boldsymbol{q}=W^Q\boldsymbol{x} \quad \boldsymbol{k}=W^K\boldsymbol{x} \quad \boldsymbol{v}=W^V\boldsymbol{x}
$$

The basic attention equation then becomes:

$$
\boldsymbol{y}_j = \sum_{i=1}^Nw_{ij}\boldsymbol{v}_i \\
w_{ij}=\boldsymbol{q}^T\boldsymbol{k}
$$

Returning to our running example, the query takes the role of the user, asking the question: _"Which movies match my taste?"_ while the key encodes the movies content and the value represents the movie itself. Using this updated definition, the computation performed by the attention operation can be visualized as follows.

<img class="img-animate" src="/images/attention/query_key_value.png">

### Attention as soft dictionary

The terms _key_ and _value_ might remind you of dictionaries used in many programming languages, where values can be stored and retrieved efficiently using a hash of the key. Indeed, such dictionaries are nothing but the digital successor of file cabinets where files (values) are stored in folders with labels (keys). Given the task to retrieve a specific file (query) you would find it by comparing the search term to all labels. Using this analogy, we can understand attention as a _soft dictionary_, returning every value to the extend that its key matches the query.

<img class="img-animate" src="/images/attention/dict.png">

### Parallelization using matrices

From the visualizations you might get the impression that attention is computed sequentially, one input at a time. Luckily, this is not the case when we move to matrix notation. All we need to do is to stack the inputs $\boldsymbol{x}\_i$ into an input matrix $X$ which we can than directly multiply with the query, key and value parameter matrices. $X$ could for example be a color image of dimension $32\times32\times3$ flattened to a matrix of shape $1024\times3$[^5].

[^5]: Remember, we are dealing with sets of elements, in this case pixels, which are feature vectors, in this case the RGB color values.

<img class="img-animate" src="/images/attention/parallel.png">

The obtained query, key and value _matrices_ $Q$, $K$ and $V$ then simply replace the individual values in the attention equation. Note how this conveniently eliminates the need for summation, an inbuilt feature of matrix multiplication, and that the softmax function can be applied immediately as all weights are computed simultaneously[^6].

[^6]: Note that $QK^T$ computes a matrix of weights where each _row_ holds the weights for the weighted sum of each output so the softmax function needs to be applied _row-wise_.

<img class="img-animate" src="/images/attention/matrix_attention.png">

There are three open questions before we can wrap this up: (1) What is $\sqrt{d\_k}$?, (2) What about _multi-head_ attention? and (3) How does this compare to simple fully-conntected layers and convolutions? Let's look at these in turn.

### Tempering with softmax

Dividing the values inside the softmax function by a scalar is known as _tempering_ and the scalar is referred to as the _temperature_. This is often beneficial as a way of calibrating the result as squishing values into zero-one range doesn't magically produce a probability distribution[^7]. On a more practical note, dividing by $\sqrt{d\_k}$ eliminates the influence of inputs with varying length[^8] and keeps values within the region of significant slope of the softmax function, thus preventing vanishing gradients and slow learning during training when attention is used in a deep learning context.

[^7]: Using the Bayesian interpretation of probabilities, good calibration means to be as confident or uncertain in a prediction as is warranted by the empirical frequency of being correct.
[^8]: The euclidean length of a vector in $\mathbb{R}^{d\_k}$ with values $v$ is $\sqrt{d\_k}v$.

### Multi-head attention

To understand the need for multiple heads when employing attention, let's consider the following example.

TODO: Add morphing sentence "Susan/Mary gave roses to Susan/Mary"

Focusing at the word _gave_, our single-headed approach will place equal attention on _Susan_ and _Mary_, as per the _distributional hypothesis_ (see below), $\boldsymbol{x}\_{susan}\approx\boldsymbol{x}\_{mary}$ which implies $\boldsymbol{q}\_{susan}\approx\boldsymbol{q}\_{mary}$ bringing us to $w\_{susan,mary}\approx w\_{mary,susan}$.

> **The distributional hypothesis:** You shall know a word by the company it keeps.

In other words, we can't discern the giver from the receiver. Another way to look at this is from the output perspective, where $\boldsymbol{y}\_{gave}$ will be identical for both permutations of Susan and Mary in the sentence. In other words, each individual output vector of attention is _permutation invariant_ with respect to the inputs while the set of outputs $Y$ is permutation _equivariant_, meaning changing the order of the inputs will change the order of the outputs in exactly the same way. Now, this is a simple problem with a simple solution: While it is difficult to discern the function of Mary and Susan in the sentence merely from the dot product similarity of their feature vectors (that is to say, using attention) it becomes trivial when looking at their _position_ in the sentence. In $A$ gave $B$, the name appearing before _gave_ is always the giver while the name after it is always the receiver, which is why architectures making use of attention, like the Transformer, extend the input representation by an additional positional component.

A more challenging problem is depicted below in the form of an ambiguous sentence we have already encountered in the [introductory article]():

<p align="center">
  <img src="/images/attention/one_head.png">
</p>

Attention weights between _it_ and all other words in the sentence are shown where darker shades correspond to greater magnitude. The meaning of _it_ is ambiguous, as it can correspond to _animal_ or _road_. Input $\boldsymbol{x}\_{it}$ has only access to a single attention weight vector $\boldsymbol{w}\_{k,it}$ though, where $k$ corresponds to all other words in the sentence (or more precisly, all of their _keys_). Thus we can only pay attention to either _animal_ or _road_ but not both. Technically we could place equal weights on both, but that doesn't help in parsing the meaning of _it_ either. The intuitive solution is to have two attention weight vectors, encoding two different meanings of our ambiguous word. This is precisely what multi-head attention achieves.

<p align="center">
  <img src="/images/attention/two_heads.png">
</p>

Multi-head attention is exactly like normal attention, just multiple times. Instead of a single $W^Q$, $W^K$ and $W^V$ matrix we now have $h$. As this would increase the computational complexity by a factor of $h$, we divide the dimensionality of the weight matrices by this factor. Our matrices $W^Q\in\mathbb{R}^{d\times d\_k}$, $W^K\in\mathbb{R}^{d\times d\_k}$ and $W^V\in\mathbb{R}^{d\times d\_v}$ become $h$ matrices $W\_i^Q\in\mathbb{R}^{d\times d\_k/h}$, $W\_i^K\in\mathbb{R}^{d\times d\_k/h}$ and $W\_i^V\in\mathbb{R}^{d\times d\_v/h}$ with inputs $\boldsymbol{x}\in\mathbb{R}^d$, queries and keys $\boldsymbol{q,k}\in\mathbb{R}^{d\_k}$ and values $\boldsymbol{v}\in\mathbb{R}^{d\_v}$. In practise, $d\_k=d\_v=d/h$ so let's ditch $d\_v$ for simplicity.

There remain two things we need to take care of. First, we don't want to apply attention sequentially $h$ times, so we stack the $h$ weight matrices for queries, keys and values respectively and then multiply them with the inputs which results in one query, key and value vector of dimension $hd\_k$ instead of $h$ each of dimension $d\_k$. Second, we want the output to have the same dimensionality as the input, so we introduce a final _output_ weight matrix $W^O\in\mathbb{R}^{hd\_k\times d}$ which transforms our intermediate outputs $\boldsymbol{z}\in\mathbb{R}^{hd\_k}$ into the final output $\boldsymbol{y}\in\mathbb{R}^{d\_k}=\boldsymbol{z}^TW^O$. That's quite a number of vectors, matrices and dimensions to juggle around in you head so hopefully the following visualization helps to clarify the concept.

TODO: Visualize multi-head matrix multiplication

### Putting attention into perspective

To wrap things up, let's try to put attention into a broader context. In particular, let's think about differences and similarities of attention to two mature building blocks of deep learning models: The fully-connected (or _dense_) and the convolutional layer. As both the fully-connected and the attention approach multiply the inputs with learnable weight matrices, you might wonder whether we even need both. While the input is projected once in the fully-connected case, attention projects it three times to obtain query, key and value representations. It then uses the similarity of the resulting queries and keys to construct outputs as weighted sums of inputs. This means that the activations of a fully-connected layer, i.e. the outputs prior to the non-linearity, are purely determined by the learned (fixed) weights during inference, while in attention, they are further modulated through their similarity. We can further see that a fully-connected layer (ignoring the non-linearity) is a special case of multi-head attention where $QK^T=W^V=I$ and $h=N$. That is to say, the _attention matrix_ $A=QK^T$ and the value weight matrix $W^V$ are identity matrices and we use one attention head per input. The output weight matrix $W^O$ then corresponds to the weight matrix of the fully-connected layer.

TODO: Maybe visualize attention as fully-connected layer

For convolutional layers, the case is similar. Again, multi-head attention can be interpreted as a generalization of discrete convolution. Given a convolutional kernel $W\in\mathbb{R}^{k\times k}$ placed on the $i^{th}$ input element we now use $h=k^2$ heads and attention weights $w_{ij}=1$ if $j=i\pm (k^2-1)/2$. This means we only consider keys which are with kernel size distance to the query where the query corresponds to the center of the kernel. With $W^V=I$ the output matrix $W^O$ then corresponds to the convolutional kernel. As the explanation is a little convoluted itself have a look at the visualization below for some intuition.

TODO: Visualize attention as convolution

## Credits

A huge thanks to Jay Alammar and Paul Bloem for their excellent blog posts and videos on the subject which formed the basis of my understanding and this article. I also want to thank all researchers, authors and creators found in the references for their great ideas and content.

---

- Explain self-attention (dot product (movie example from "Transformers from Scratch", recommender systems, matrix factorization), illustrations from "Illustrated Transformer")
- Use visuals from [1, 6, 9, 20] and notation from [14]
- Use code? (Feynman: "What I cannot create I do not understand")
- Explain relation to MLP: How is weighing by input (self-attention score) different from weighing by FC weight matrix? Example: Hard-coding connection strength to verbs, nouns and articles (only works with the exact same sentence structure) vs. ordering the input (self-attention) and passing it to specialized parts of the model afterwards
- Explain relation to CNN: convolution as multi-head self-attention with identity key, query and value matrices, difference between weighing input locations with (kernel) weights (only change during training) and modulating those connections during inference through attention (reduced redundancy of channels: +-45° edge detector can become one?)

## Notes

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
- Dot-product self-atttention: softmax(key.query)\*value
- Vectorize: $W=softmax(XX^T)$ (row-wise softmax); $Y^T=WX^T$
- $softmax(KQ^T)$: A probability distribution over keys with modes where key and query are similar
- Diagonal of W is largest in this basic setup; no parameters
- Linear computation between X and Y (strong gradients)
- Same distance from every input to every output (pair-wise self-attention): Different form RNNs where distance grows
- Self-attention is a set model not a sequence model: Sequential information needs to be added on top (positional embedding)
- Permutation equivariant: p(sa(X)) = sa(p(X))
- Power of dot-product: Respects magnitude and sign (recommender systems)
- Scaled dot-product by sqrt(k) where k is the number of elements: Prevents vanishing gradient in softmax
- Attention as "soft" dict (key, value): Every key matches the query to some extent and a mixture of values is returned (image from [20])
- Liner transformations on X: K, Q, V allow the key, query and value to play different roles
- Multi-head self-attention (multiple self-attention operations applied in parallel): A token can relate to another in more than one way
- Down-project input with one linear transformation per head, concatenate outputs, apply a single linear transformation
- Same number of parameters: Linear transforms/output is simply cut into head-sized pieces
- Transformer: A model that primarily uses self-attention to model input element dependencies
- Transformer block: input -> layer norm -> self-attention -> residual -> layer norm -> feed-forward -> residual -> output
- Layer norm similar to batch norm, except that the normalization is along the input feature dimension instead the batch dimension
- In autoregressive models, self-attention weights $X^TX$ upper diagonal matrix is set to -infinity (becomes zero after softmax) to prevent lookahead in time)
- Position embedding: Each element gets an ID (learned or hand-crafted); fixed maximum length
- Position encodings: Each element gets an ID of multiple values from overlayed frequencies (e.g. sine/cosine)
- Relative position: Distance to current element
- Transformer: Attention is all you need (no other bells and whistles required), translation model (encoder-decoder)
- BERT: Spiritual successor to ELMo (unsupervised pre-training, supervised finetuning), encoder only
- Trained on two tasks: Masked/corrupted word prediction (bi-directional language model task), sentence entailment (classification)
- GPT-2: Autoregressive model (generative); can produce long-range coherence
- Encoder-decoder architecture used in translation to cross-attend between input and output (e.g. to deal with different word order)
- Multi-head attention like convolution
- RNN with attention: Passes all hidden states from the encoder to the decoder instead of just the last, compute attention score between current decoder hidden state and all encoder hidden states to choose the most relevant
- Attention as memory: Stores context, access it through association
- Transformer uses shared MLP on each hidden state after self-attention (in practice, vectorized with input vectors stacked into a matrix as in PiontNet)
- Multiple heads help to disentangle ambiguous information of input elements, i.e. pay attention to different parts of a sentence for different meanings of a word or different points of reference
- MLP after multi-head self-attention expects a single matrix of output vectors, not matrices, so another linear projection with an output weight matrix is needed (this is the one that corresponds to the CNN kernel)
- Cross-/encoder-decoder attention: key, query come from encoder, value from previous decoder
- Greedy decoding in NMT: Only use the highest probability word from current output (time step) to predict next
- Beam search: Use the two highest probability words from current output (beam size 2) and run the model twice, pick the better continuation
- RNN steps through time sequentially, transformers in parallel
- Self-attention is like learned pre-processing (filtering) prior to representation learning (i.e. like convolution imposes a geometric prior before kernel weights are trained)
- Self-attention is a universal context prior: By adding it to a model, one implicitly tells it to _first_ select elements with high utility, and _then_ learn to solve the problem using the selection
- It performs well everywhere because it is general and _required_ everywhere (every problem solver needs to select relevant context)
- The new representation of an element after self-attention is a weighted average of all prior representations of all elements; The subsequent MLP only lifts it into higher dimensional feature space to make it linearly separable
- Encoding is done in parallel, decoding is still sequential, because it relies on its own output of the previous step
- Dot product as similarity measure: cares about both angle (direction) between and magnitude (length) of two vectors
- Works for words in embedding space (distributional hypothesis) but not in input space (words with similar characters have no semantic similarity)
- correlation = cosine similarity
- covariance = dot product similarity
- Encoder-decoder architecture needed to handle differing input/output sequence length?: No, only output length <= max input length required
- Because RNNs are sequential but the entire context (e.g. sentence) is needed to produce the output (e.g. translation), the input sequence first needs to be encoded entirely (encoder) to then be decoded. Not needed with Transformers as they can encode the entire input in parallel)
- Using convolutions or RNNs, the distance of two inputs in the input spaces determines their distance in feature space, thus long-range dependencies are treated inherently differently, which is usually not semantically warranted
- The transformer sacrifices resolution (due to averaging of attention-weighted positions) for constant distance which is mitigated through the multi-head design
- Self-attention in original paper = scaled dot-product attention
- Dot-product and additive attention [18] have similar theoretical complexity, but dot-product is much faster due to highly optimized matrix multiplication on GPUs
- Three types of attention in original transformer: encoder-decoder, self, decoder (masked)
- $Y=softmax(KQ^T)V$ is permutation _equivariant_ for each element in $Y$ and permutation _invariant_ for each element in $y$ (e.g. bgr=rgb)
- Without K, Q, V, self-attention produces outputs entirely determined by the input embeddings (word2vec representation of mary and susan is probably quite similar wrt "gave" so the output will be ambiguous wrt who is doing the giving and who is receiving, i.e $w_{susan,gave}=w_{mary,gave}$)
- Adding Q, K, V, self-attention can produce $w_{susan,gave}=q_{susan}^Tk_{gave}>w_{mary,gave}=q_{mary}^Tk_{gave}$ but:
- Without multiple heads, "mary gave roses to susan" = "susan gave roses to mary" wrt. $y_{gave}$, because $y_i=\sum_jw_{ij}x_j$ in self-attention is permutation _invariant_
- This simple problem can also be solved by using positional embeddings, but "The animal didn't cross the road because it was too tired/wide." can't (meaning of $y_{it}$ can't be parsed by position)
- Multiple heads: Split dimensionality of Q (and K, V) from d x q into d x q/h, compute $Y_1...Y_H$ with $Q_1...K_H$ (and K, V), concat Ys, project back to d x N with W_O (paper p. 5)
- With average pooling in the end (e.g. for classification) the Transformer becomes permutation invariant
- Decoder-only used as synonym for autoregressive model with masking (generative model), encoder only as synonym for classification model (language model)
- So far, performance is limited by hardware not model design
- GPT-X is decoder-only (language model, masked self-attention: left-to-right context only) and thus can do NMT (no encoder required)
- BERT is encoder-only (predict masked/wrong words: Cloze task, self-attention: bi-directional context)
- Attention originally used as connection between encoder and decoder
- "Time" step in RNNs is one token
- Attention: A mapping from query to a set of key-value pairs; the output is their weighted sum
- Weights are computed by a compatibility function between query and key
- Self-attention has no parameters, they are introduced in the multi-head design, but an activation function (softmax)
- $1/\sqrt(d_k)$ scaling: Large values of dot product push the softmax in regions of small gradient [20]; dimensionality of query, key (d_k) would influence output, scaling by $1/\sqrt(d_k)$ removes this [9]
- Multi-head attention has parameters, but no non-linearity
- The shorter the path between input and corresponding output elements (in computation steps), the easier it is to learn their relation (due to vanishing gradients)
- Using RNNs, the distance grows linearly with length and with length/kernel size for convolutions ($\log_k(n)$ for dilated convs); Using self-attention it's constant
- Depth-wise separable convolutions have the same complexity as self-attention + shared MLP
- Sequence length is usually shorter than representation dimensionality (for sentence level problems with word-piece or byte-pair encoding)
- Initial use of attention: called soft-attention (see soft dict), used in encoder-decoder configuration (not self-attention), a MLP to compute attention weights, used as a way to take burden from fixed-length encoder output, need to use forward and backward LSTM to capture bi-directional context, focused on alignment of source and target
- Summation of weighted values as _expected correspondence_
- Alternatives to dot-product attention: _general_ ($q^TWk$), _mlp_
- Probabilistic perspective: Maximize conditional probability of target set given the input set (likelihood)
- The big idea of attention: Instead of a fixed size context vector, use current input dependent context
- Multi-head like features maps in CNN (feature/channel dimension)
- Transformers are graph neural networks

## Code & References

|      |                                                                                                                                         | Check |
| :--: | :-------------------------------------------------------------------------------------------------------------------------------------- | :---: |
|      |                                                                                                                                         |       |
| [1]  | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer)                                                       |   x   |
| [2]  | [The Narrated Transformer](https://www.youtube.com/watch?v=-QH8fRhqFHM)                                                                 |   x   |
| [3]  | [Transformer Neural Networks](https://www.youtube.com/watch?v=TQQlZhbC5ps)                                                              |   x   |
| [4]  | [Bert Neural Network](https://www.youtube.com/watch?v=xI0HHN5XKDo)                                                                      |   x   |
| [5]  | [Attention in Neural Networks](https://www.youtube.com/watch?v=W2rWgXJBZhU)                                                             |   x   |
| [6]  | [Self-Attention](https://www.youtube.com/watch?v=KmAISyVvE1Y)                                                                           |   x   |
| [7]  | [Transformers](https://www.youtube.com/watch?v=oUhGZMCTHtI)                                                                             |   x   |
| [8]  | [Famous Transformers](https://www.youtube.com/watch?v=MN__lSncZBs)                                                                      |   x   |
| [9]  | [Transformers from Scratch](http://peterbloem.nl/blog/transformers)                                                                     |   x   |
| [10] | [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)                                                      |   x   |
| [11] | [Visualizing a NMT Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention) |   x   |
| [12] | [Transformer (Google AI Blog)](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)                                 |   x   |
| [13] | [Attention is all you need (video)](https://www.youtube.com/watch?v=rBCqOTEfxvg)                                                        |   x   |
| [14] | [Attention Is All You Need (paper)](https://arxiv.org/abs/1706.03762)                                                                   |   x   |
| [15] | [What are Transformers?](https://www.youtube.com/watch?v=XSSTuhyAmnI)                                                                   |   x   |
| [16] | [Illustrated Guide to Transformers](https://www.youtube.com/watch?v=4Bdc55j80l8)                                                        |   x   |
| [17] | [An Introduction to Attention](https://wandb.ai/authors/under-attention/reports/An-Introduction-to-Attention--Vmlldzo1MzQwMTU)          |   x   |
| [18] | [Origins of Attention I](htps://arxiv.org/abs/1409.0473)                                                                                |   x   |
| [19] | [Origins of Attention II](https://arxiv.org/abs/1508.04025)                                                                             |   x   |
| [20] | [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2)                                                                    |   x   |
| [21] | [Spatial Attention in Deep Learning](https://arxiv.org/abs/1904.05873)                                                                  |   x   |

$$
$$
