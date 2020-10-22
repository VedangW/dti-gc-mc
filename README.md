# Graph Convolutional Matrix Completion for Drug-Target Interaction Prediction

This is the Tensorflow based implemention of Graph Convolutional Matrix Completion, forked from [Rianne van den Berg's implementation](https://github.com/riannevdberg/gc-mc), applied to the problem of drug-target interaction prediction.

Key changes include:
- Changed sampling strategy, sincce a drug-target interaction matrix is highly sparse.
- Some changes in the model for experimentation.

Original research paper:
Rianne van den Berg, Thomas N. Kipf, Max Welling, [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263) (2017)
