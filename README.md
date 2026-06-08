# Vanilla-Transformer-Architecture-From-Scratch
In this project, I am trying to implement the architecture of the vanilla transformer in python, and pass a single batch of few sequences of tokens through it to understand how each component works!

![Transformer](ModalNet-21.png)

The link for the landmark paper is https://arxiv.org/abs/1706.03762

The main components of this architecture are

- Token Embeddings
- Three variants of multi-head attention (plain, masked and cross attention) 
- Add & Norm (Layer Normalization)
- Feed Forward Block (A multi layer perceptron)
- Linear Projections
- Softmax

Essentially, the two main components of transformer are Encoder and Deecoder blocks, but once we have the above described components ready, we can easily obtain the encoder and decoder blocks. 



