---
layout: post
title: Attention
abstract: Yet another article on attention? Yes, but this one is both annotated and illustrated focusing on attention itself instead of the architecture making it famous. After all, attention is all you need, not Transformers.
tags: [attention, transformer, context]
category: learnig
mathjax: true
jquery: true
plotly: true
slideshow2: true
slow: false
time: 0
words: 0
---

# {{ page.title }}

You might be wondering if this article can possible contain anything new for you. You already studied the [illustrated]() and [annotated]() Transformer, the [original paper]() and everything from [GPT-1-2-3]() to [BERT]() and beyond. Well, if you have, it could indeed be that there is nothing _fundamentally_ new for you to be found here, but the goal to weave all this information into one coherent story and provide further context across dimensions and domains with a focus on attention itself instead of the scaffolding erected around it referred to as _The Transformer_ has potential to further clarify and solidify some of these truly interesting and general concepts.
There story is told in three acts: I. Why, II. How, III. Where. Simple.

## Why? Conceiving context

It all begins with a little entity making up most of the (digital) world around you. It takes many names some calling it _word_, _pixel_ or _point_, but we will simply call it _element_. Our little element is secretive, revealing almost nothing about itself in isolation. In that regard, it is like its sibling in the real world, the atom. Both are atomic[^1]. It has emergent properties though: Throw a couple of thousand of them together and you get a story, an image, a 3D model. What has changed? The Context.

> **Context:** The circumstances that form the setting for an event, statement, or idea, and in terms of which it can be fully understood.

Let's look at a couple of examples. The simplest (an therefore the one we will see most frequently throughout the article) is the word. Try to guess the meaning of the word below, then hover over it with your cursor or click to reveal the context:

<img class="img-animate" src="/images/attention/bank.png">

Did you guess the meaning correctly? Or was it the financial institution or place to sit? The point is, of course, that you couldn't have known without the context of the entire sentence, as many words are ambiguous. It doesn't stop there though. Even the sentence is ambiguous if your goal is to determine the book title or author who wrote it. To do so, you might need a paragraph, a page or even an entire chapter of context. In machine learning lingo, such broad context is commonly called a _long-range dependency_. Here is another one. Pay attention to the meaning of the word _"it"_:

![](/images/attention/it.gif)

Seeing _"tired"_, we know _it_ must refer to the animal, as roads are seldom so while it's the opposite for _"wide"_[^2].

Below, there are two more examples of increasing dimensionality (use the little arrows to switch between them). While sentences can be interpreted as one-dimensional sequences of word-elements, an image is a two-dimensional grid of picture-elements (pixels) and a 3D model can be represented by a cloud of point-elements[^3] (or volumetric-elements: voxels). You will notice that you can't discern what is represented by the closeup view of the individual elements but when zooming out (using the "Zoom out" buttons and your mousewheel or fingers) the interpretation becomes trivial.

<br/>
<div id="slideshow1" class="slideshow-container">
  <div class="mySlides fade">
    {% include figures/image_zoomed.html %}
  </div>

  <div class="mySlides fade">
    {% include figures/happy_buddha.html %}
  </div>

<a class="prev" onclick="plusSlides(-1, this.parentNode)">&#10094;</a>
<a class="next" onclick="plusSlides(1, this.parentNode)">&#10095;</a>

</div>
<br/>

Again, context doesn't stop there. To correctly place a pixel as belonging to, say, an eye, you need the surrounding pixels making up the eye. To place the eye as coming from an adult or a child you make use of the information stored in the pixels around the eye. Such inference can potentially go on indefinitely, but it's usually restricted by the size of the depicted scene or the resolution of the image. Okay, you might think, so is more information always better? No.

![](/images/attention/knowledge.gif)

Finding the hidden information in the image above is trivial if the surrounding context is removed (to be precise, it's not the _absence_ of context, as all pixels are still there, but the _contrast_ between signal and noise, percieved as difference between gray and colored pixels). Clearly, it's not a simple as having no context at all or all of it but rather which portion of the provided information we pay _attention_ to.

[^1]: Not really of course, words can be divided into letters, atoms in particles, but let's ignore that.
[^2]: This, and many more of these (deliberately) ambiguous sentences can be found in the _Winograd schema challenge_.
[^3]: Also known as a _point cloud_. Take a look at the previous articles on learning from 3D data for other representations.

## How? Context across dimensions and domains

Now that you are hopefully convinced that context is an important concept across domains, let's start this section off by investigating how how researchers have dealt with it prior to the advent of attention. First up are sequence data in the form of written and spoken language. Then we will look at images and 3D data formats in turn.

### Context without attention

For a long time, the predominant method used to model natural language was the _Recurrent Neural Network_ (RNN), first in a basic fully-connected flavor and later using _Long-Short-Term Memory_ (LSTM) and _Gated Recurrent Units_ (GRU). In this paradigm, context is accumulated over time, one element (word) after another, allowing the model to reference potentially indefinitely into the past. Potentially, because in reality it turns out that the memory is rather short and plagued by vanishing and exploding gradients, a problem addressed to some extend by the LSTM and GRU variants. The recurrent nature, while interesting, also requires sequential computation, resulting in slow training and inference. Additionally, the model can't look into the future,[^4] requiring a complicated encoder-decoder design, first aggregating the entire context of the input to provide the decoder with the necessary global context,[^5] a point we will return to in the last section of this article.

<div style="text-align: center">
<figure style="width: 90%; display: inline-block;">
  <img src="/images/attention/rnn_unrolled.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>An unrolled RNN:</b> Input elements $\boldsymbol{x}_t$ are processed sequentially while context is retained in the hidden states $\boldsymbol{h}_t$. [1]</figcaption>
</figure>
</div>

[^4]: Except for bi-directional RNNs which read the sequence from left and right.
[^5]: An example being machine translation: The input sequence (a sentence in English) is first encoded from the first to the last element (word) and then decoded sequentially to produce the translation (the sentence in French).

The shortcomings of RNNs motivated the look into alternatives, one of which was found in a revered companion: the convolution. At first glance, this might seem like a strange choice, considering convolutions as almost synonymous with locality.
However, there are at least two tricks to aggregate long-range dependencies using convolutions. The first one is to simply stack them. Maybe due to their prevalence in image processing, the range covered by a convolution is called its _receptive field_ and through stacking, it can be grown.

<div style="text-align: center">
<figure style="width: 90%; display: inline-block;">
  <img width="400" style="margin-left: auto; margin-right: auto;" src="/images/attention/1d_conv.png">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Receptive field:</b> The receptive field (red) is the range of context aggregated into the current representation. It increases with the number of stacked convolutional layers. [1]</figcaption>
</figure>
</div>

Now, for large inputs (a long text, a high resolution image, a dense point cloud), this simple way of increasing the receptive field size is inefficient, because we need to stack many layers which bloats the model. A more elegant way is to use _strided_ convolutions, where the convolutional kernel is moved more than a single element, or _dilated_ (atrous) convolutions, where the kernel weights are scattered across the input with perceptive holes (French: _trous_) in between. As we might miss important _in between_ information with this paradigm we can again stack multiple such convolutions with varying strides or dilation factors to efficiently cover the entire input.

<div style="text-align: center">
<figure style="width: 90%; display: inline-block;">
  <img src="/images/attention/wavenet.gif">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>WaveNet:</b> Using dilated convolutions, long sequences can be processed efficiently while retaining a large receptive field. [2]</figcaption>
</figure>
</div>

Moving to the image domain, there is no fundamentally new idea here as vision models still largely rely on convolutions with similar characteristics as introduces above, the only change being the added second dimension.

<div style="text-align: center">
<figure style="width: 90%; display: inline-block;">
  <img src="/images/attention/dilated.gif">
  <figcaption style="text-align: left; line-height: 1.2em;"><b>Dilation in 2D:</b> Concepts like strided and dilated convolutions work identically in one, two or three dimensions. [3]</figcaption>
</figure>
</div>

Adding a third dimension, things get more interesting again, as the computational complexity of convolutions becomes a major problem. While they can be used successfully, the input usually needs to be downsampled considerably prior to their application. Another approach is to use an element-wise feed-forward neural network[^6]. This approach is extremely efficient, but doesn't consider _any_ context. To resolve this, context aggregation is performed by an additional process like _Farthest Point Sampling_, _k Nearest Neighbor_ search or _Ball Queries_. One exception is the _Graph Neural Network_. As the name implies, it works on graphs as input (either dynamically computed or static ones as found in triangle meshes) and can leverage graph connectivity for context information. I've written an entire mini-series on learning from various 3D data representations which I invite you to check out if the above seems inscrutable.

[^6]: Sometimes referred to as _shared MLP_ (Multi-Layer Perceptron), which in the end boils down to a 1x1 convolution as discussed [here](https://hummat.github.io/learning/2020/10/29/one-by-one-conv.html).

<br/>
<div id="slideshow2" class="slideshow-container">
  <div class="mySlides fade">
    {% if page.slow %}
      {% include /figures/bunny_with_spheres.html %}
    {% else %}
      <img src="/images/bunny.png">
    {% endif %}
    <div class="text"><b>Point context:</b> Defining context regions using farthest point sampling and ball queries.</div>
  </div>

  <div class="mySlides fade">
    {% if page.slow %}
      {% include figures/3d_conv.html %}
    {% else %}
      <img src="/images/3d_conv.png">
    {% endif %}
    <div class="text"><b>Convolutions in 3D:</b> Adding a dimension drastically increases the computational burden of convolutions, making them cumbersome in the 3D domain.</div>
  </div>

  <div class="mySlides fade">
    {% if page.slow %}
      {% include /figures/graph.html %}
    {% else %}
      <img width="220" style="margin-left: auto; margin-right: auto; display: block;" src="/images/mesh.png">
    {% endif %}
    <div class="text"><b>Graph context:</b> A mesh can be interpreted as a graph where context is expressed through connectivity.</div>
  </div>

<a class="prev" onclick="plusSlides(-1, this.parentNode)">&#10094;</a>
<a class="next" onclick="plusSlides(1, this.parentNode)">&#10095;</a>

</div>
<br/>

### <Insert some pun using "attention">

We have finally arrived at the heart of this article: A detailed explanation of the attention mechanism. This involves some math which will subsequently be visualized to build an intuitive understanding. The attention equation makes use of scalars, vectors and matrices. We will use lowercase letters $x$, bold lowercase letters $\boldsymbol{x}$ and uppercase letters $W$ for these entities respectively. A great way to visualize scalars, vectors and matrices is to represent them by colored squares, where the number of squares represents the dimensionality and the color vibrancy of each square represents the magnitude of the value at this position from small, bright values to large, dark values. This is summarized below.

![](/images/attention/notation.gif)

Making use of this notation, the basic attention equation can be written as

$$
\boldsymbol{y}_j = \sum_{i=1}^Nw_{ij}\boldsymbol{x}_i
$$

So what's happening here? In a nutshell: The output of the attention operation $\boldsymbol{y}\_j$ is a weighted sum of the inputs $\boldsymbol{x}\_i$. The inputs are the set elements introduced previously, for example words, pixels or points in 3D space. They are usually vectors, where the dimensions, commonly called input features, can represent various properties like RGB color channels or XYZ Euclidean coordinates. Words are somewhat special in this regard, as they don't have any intrinsic features and are therefore commonly represented by a large vector with values encoding their semantics computed from their co-occurrence with other words called an _embedding_[^7].

[^7]: Search for _"word2vec"_ if you want further details. I recommend [this blog post](https://jalammar.github.io/illustrated-word2vec).

What are those weights $w\_{ij}$ then and how are they determined? Though we employ attention in the deep learning domain, in it's most basic form, it actually doesn't feature any learnable parameters. Instead, the weights are computed from the inputs using the dot product as similarity measure as $w\_{ij}=\boldsymbol{x}\_i^T\boldsymbol{x}\_j$. To understand how this works and why it makes sense, let's take a look at the visualization below.

<img class="img-animate" src="/images/attention/dot_product.png">

The two vectors $\boldsymbol{u}$ and $\boldsymbol{m}$ represent the movie preferences of a user and the movie content respectively with three features each. Now, taking the dot product, we get a score representing the match between user and movie. Note that this takes into account both the magnitude of the values, as multiplying large values results in a large increase of the score, as well as the sign. Imagine a scale between $-1$ and $1$ where the former represent strong dislike (or weak occurance in case of the movie vector) and the latter a strong inclination (or strong occurance). Now, given a user who dislikes action and a movie with little action, the score will still be high as both negative signs cancel out.

To make those weights more interpretable and prevent the output from becoming excessively large, we can normalize all weights $\boldsymbol{w}\_j=\{w\_{ij}\}\_{i=1}^N$ to fall between $0$ and $1$ using the _softmax_ function which is reminiscent of the its application on the logits in classification tasks. We can think about the resulting normalized weight vector as a categorical probability distribution over the inputs where high probability is assigned to inputs with strong similarity[^8]. Once all weights are computed, we multiply them with their corresponding inputs and sum them up to obtain the $j^{th}$ output which, as stated before, is simply a weighted sum of the inputs. The computation graph below visualizes the entire process.

[^8]: Keep in mind though, that this is not a real probability distribution where the probability values correspond to empirical frequencies but rather a miscalibrated approximation.

<img class="img-animate" src="/images/attention/attention.png">

As in the dot product example, think of the transposed vectors (the horizontal ones) as individual users and the vertical ones as movies. The outputs would then represent something akin to an ideal movie, one for each user, stitched toghether from the three individual movies on offer. But wait, we aren't actually dealing with different entities like users and movies in this basic formulation of attention, as all three vectors involved, two of which are used to compute the weight and the remaining one as part of the weighted sum, come from the same set $X$. This also means that output $\boldsymbol{y}\_1$ will be dominated by input $\boldsymbol{x}\_1$ as the dot product between two identical vectors is usually large (after all, they are as similar as it gets). To deal with these issues, we can try to make the attention computation more versatile and expressive through the introduction of three learnable parameter matrices $W^Q$, $W^K$ and $W^V$. Through multiplication with the inputs $\boldsymbol{x}$ we obtain three distinct vectors for the three roles on offer namely the <span style="color: #AB63FA; font-weight: bold;">query $\boldsymbol{q}$</span>, <span style="color: #FECB52; font-weight: bold;">key $\boldsymbol{k}$</span> and <span style="color: #00CC96; font-weight: bold;">value $\boldsymbol{v}$</span>.

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

The terms _key_ and _value_ might remind you of dictionaries used in many programming languages, where values can be stored and retrieved efficiently using a hash of the key. Indeed, dictionaries are nothing but the digital successor of file cabinets where files (values) are stored in folders with labels (keys). Given the task to retrieve a specific file (query) you would find it by comparing the search term to all labels. Using this analogy, we can understand attention as a _soft dictionary_, returning every value to the extend that its key matches the query.

<img class="img-animate" src="/images/attention/dict.png">

From the visualizations you might get the impression that attention is computed sequentially, one input at a time. Luckily, this is not the case when we move to matrix notation. All we need to do is to stack the inputs $\boldsymbol{x}\_i$ into an input matrix $X$ which we can than directly multiply with the query, key and value parameter matrices.

![](/images/attention/parallel.png)

The obtained query, key and value _matrices_ $Q$, $K$ and $V$ then simply replace the individual values in the attention equation. Note how this conveniently eliminates the need for summation, an inbuilt feature of matrix multiplication, and that the softmax function can be applied immediately as all weights are computed simultaneously.

![](/images/attention/matrix_attention.png)

There are three open questions before we can wrap this up: (1) What is $\sqrt{d\_k}$?, (2) What about _multi-head_ attention? and (3) Is there any relationship to convolutions? Let's look at these in turn.

Dividing the values inside the softmax function by a scalar is known as _tempering_ and the scalar is referred to as the _temperature_. This is often beneficial as a way of calibrating the result as squishing values into zero-one range doesn't magically produce a probability distribution[^9]. On a more practical note, dividing by $\sqrt{d\_k}$ eliminates the influence of inputs with varying length and keeps values within the region of significant slope of the softmax function, thus preventing vanishing gradients and slow learning during training when attention is used in a deep learning context.

[^9]: Using the Bayesian interpretation of probabilities, good calibration means to be as confident or uncertain in a prediction as is warranted by the empirical frequency of being correct.

To understand the need for multiple heads when employing attention, let's consider the following example.

- Explain self-attention (dot product (movie example from "Transformers from Scratch", recommender systems, matrix factorization), illustrations from "Illustrated Transformer")
- Use visuals from [1, 6, 9, 20] and notation from [14]
- Use code? (Feynman: "What I cannot create I do not understand")
- Explain relation to MLP: How is weighing by input (self-attention score) different from weighing by FC weight matrix? Example: Hard-coding connection strength to verbs, nouns and articles (only works with the exact same sentence structure) vs. ordering the input (self-attention) and passing it to specialized parts of the model afterwards
- Explain relation to CNN: convolution as multi-head self-attention with identity key, query and value matrices, difference between weighing input locations with (kernel) weights (only change during training) and modulating those connections during inference through attention (reduced redundancy of channels: +-45° edge detector can become one?)

## Where?

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

| [Code](/url/to/notebook.ipynb) | [![Binder](https://mybinder.org/badge_logo.svg)](/url/to/binder/notebook.ipynb)                                                         | Check |
| :----------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------- | :---: |
|                                |                                                                                                                                         |       |
|              [1]               | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer)                                                       |   x   |
|              [2]               | [The Narrated Transformer](https://www.youtube.com/watch?v=-QH8fRhqFHM)                                                                 |   x   |
|              [3]               | [Transformer Neural Networks](https://www.youtube.com/watch?v=TQQlZhbC5ps)                                                              |   x   |
|              [4]               | [Bert Neural Network](https://www.youtube.com/watch?v=xI0HHN5XKDo)                                                                      |   x   |
|              [5]               | [Attention in Neural Networks](https://www.youtube.com/watch?v=W2rWgXJBZhU)                                                             |   x   |
|              [6]               | [Self-Attention](https://www.youtube.com/watch?v=KmAISyVvE1Y)                                                                           |   x   |
|              [7]               | [Transformers](https://www.youtube.com/watch?v=oUhGZMCTHtI)                                                                             |   x   |
|              [8]               | [Famous Transformers](https://www.youtube.com/watch?v=MN__lSncZBs)                                                                      |   x   |
|              [9]               | [Transformers from Scratch](http://peterbloem.nl/blog/transformers)                                                                     |   x   |
|              [10]              | [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)                                                      |   x   |
|              [11]              | [Visualizing a NMT Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention) |   x   |
|              [12]              | [Transformer (Google AI Blog)](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)                                 |   x   |
|              [13]              | [Attention is all you need (video)](https://www.youtube.com/watch?v=rBCqOTEfxvg)                                                        |   x   |
|              [14]              | [Attention Is All You Need (paper)](https://arxiv.org/abs/1706.03762)                                                                   |   x   |
|              [15]              | [What are Transformers?](https://www.youtube.com/watch?v=XSSTuhyAmnI)                                                                   |   x   |
|              [16]              | [Illustrated Guide to Transformers](https://www.youtube.com/watch?v=4Bdc55j80l8)                                                        |   x   |
|              [17]              | [An Introduction to Attention](https://wandb.ai/authors/under-attention/reports/An-Introduction-to-Attention--Vmlldzo1MzQwMTU)          |   x   |
|              [18]              | [Origins of Attention I](htps://arxiv.org/abs/1409.0473)                                                                                |   x   |
|              [19]              | [Origins of Attention II](https://arxiv.org/abs/1508.04025)                                                                             |   x   |
|              [20]              | [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2)                                                                    |   x   |
|              [21]              | [Spatial Attention in Deep Learning](https://arxiv.org/abs/1904.05873)                                                                  |   x   |

$$
$$
