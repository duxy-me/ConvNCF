# Outer Product-based Neural Collaborative Filtering

Experiments codes for paper "Outer Product-based Neural Collaborative Filtering".

# Requirements

- Tensorflow 1.7
- numpy, scipy

# Directories

- Data. Training and testing data.
    - yelp.train.rating. Rating of training data.
    - yelp.test.rating. Rating of testing data.
    - yelp.test.negative. 1000 testing samples for each user. (0,32) means this row is for user 0 and the positive test item is 32.
- Dataset.py. Module preprocessing data.
- saver.py. Module saving parameters.
- MF_BPR.py. MF model with BPR loss.
- ConvNCF.py. Our model.

# How to train ConvNCF

0. decompress the data files.
    ```cd Data;gunzip *```

1. Pretrain the embeddings using MF_BPR with

    ```python MF_BPR.py```

2. Train ConvNCF with pretrained embeddings

    ```python ConvNCF.py --pretrain=1```

3. You may further tune the model using dropout layers with extensive arg `--keep=xx`

