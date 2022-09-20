# CoNet

The code for our "CoNet: Collaborative Cross Networks for Cross-Domain Recommendation" paper published at CIKM 2018 


## Deep Transfer Learning Framework 
![Framework](/image/TransDL.png "DeepTL")


## Readme

There are three folders:

### Data
1. "./data". The data is INCOMPLETE since some servers limit the size of an email. As a result, I put one example for each data file.

### The CoNet model
2. "./CoNet_mtl_cross_1223hid". This is the CoNet model described in the Section 4.2 in our paper. There, 'mtl'=multitask learning, 'cross'=cross Connections Unit, '1223hid'=cross units enforced between hidden layer 1 and hidden layer 2, and enforced between hidden layer 2 and hidden layer 3. See the illustration Figure 2 in our paper. Tune these hyperparameters on your own datasets.

### SCoNet, The Sparse Variant of CoNet
3. "./SCoNet_mtl_lasso_cross_1223". This is the SCoNet model described in the Section 4.3 in our paper. There, 'lasso'=l1-norm penalty describe in Eq. (9) in our paper.



## See More... 

Our [project page](https://njuhugn.github.io/research-conet.html)