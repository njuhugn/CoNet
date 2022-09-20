# CoNet

The code for our "<strong>CoNet: Collaborative Cross Networks for Cross-Domain Recommendation</strong>" paper published at [CIKM 2018](https://njuhugn.github.io/research-conet.html)


## Deep TL Framework 
![](/image/TransDL.png)


## Data & Model

### Data
"<strong>./data</strong>". The cross-domain datasets split into Train/Valid. I put one example for each data file since the whole dataset consumes much storage. The original Amazon.com data can be downloaded [here](http://snap.stanford.edu/data/web-Amazon.html) and the other Cheetah Mobile cannot be publicly available due to privacy (send email to us).

### The CoNet model
"<strong>./CoNet_mtl_cross_1223hid</strong>". This is the CoNet model described in the Section 4.2 in our paper. Here, 'mtl'=multitask learning, 'cross'=cross Connections Unit, '1223hid'=cross units enforced between hidden layer 1 and hidden layer 2, and enforced between hidden layer 2 and hidden layer 3. See the illustration Figure 2 in our paper. Tune these hyperparameters on your own datasets.

### SCoNet, The Sparse Variant of CoNet
"<strong>./SCoNet_mtl_lasso_cross_1223</strong>". This is the SCoNet model described in the Section 4.3 in our paper. Here, 'lasso'=l1-norm penalty describe in Eq. (9) in our paper.


## Runtime
Our methods are implemented using TensorFlow. For the training time, our models spend about 100 seconds per epoch using one Nvidia TITAN Xp GPU. As a reference, it is 70s for MLP and 90s for CSN models.

## Acknowledgement

Please cite the following paper if our code+paper helps your research.

```
@@inproceedings{hu2018conet,
  title={Conet: Collaborative cross networks for cross-domain recommendation},
  author={Hu, Guangneng and Zhang, Yu and Yang, Qiang},
  booktitle={Proceedings of the 27th ACM international conference on information and knowledge management},
  pages={667--676},
  year={2018}
}
```

## See More... 

...Our [project page](https://njuhugn.github.io/research-conet.html)