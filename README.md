## Chinese Word Segmentation Using Neural Architectures

This is a Tensorflow implementation of the named eneity recognition model described in the [paper] (https://arxiv.org/abs/1603.01360). The original Theano implementation could be found [here] (https://github.com/glample/tagger)
.

### Model

- Word Embedding (initialized using [word2vec](https://code.google.com/archive/p/word2vec/))
- Birectional LSTM
- CRF

### Data

The model is trained on People's Daily 2014. The data is included in the repo.

### Train

The model can be trained by running:

```
sh run.sh
```

### References

- https://arxiv.org/abs/1603.01360
- https://github.com/glample/tagger
- https://github.com/koth/kcws