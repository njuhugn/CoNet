# CoNet

The code for our "<strong>CoNet: Collaborative Cross Networks for Cross-Domain Recommendation<strong>" paper published at [CIKM 2018](./conet-cikm18.pdf)


## Deep Transfer Learning Framework 
![Framework](/image/TransDL.png "DeepTL")


## Readme

There are three folders:

### Data
"./data". The cross-domain datasets split into Train/Valid. I put one example for each data file since the whole dataset consumes much storage. The original Amazon.com data can be downloaded [here](http://snap.stanford.edu/data/web-Amazon.html) and the other Cheetach Mobile cannot be publicly available due to privacy (send email to us).

### The CoNet model
"./CoNet_mtl_cross_1223hid". This is the CoNet model described in the Section 4.2 in our paper. There, 'mtl'=multitask learning, 'cross'=cross Connections Unit, '1223hid'=cross units enforced between hidden layer 1 and hidden layer 2, and enforced between hidden layer 2 and hidden layer 3. See the illustration Figure 2 in our paper. Tune these hyperparameters on your own datasets.

### SCoNet, The Sparse Variant of CoNet
"./SCoNet_mtl_lasso_cross_1223". This is the SCoNet model described in the Section 4.3 in our paper. There, 'lasso'=l1-norm penalty describe in Eq. (9) in our paper.



## See More... 

Our [project page](https://njuhugn.github.io/research-conet.html)