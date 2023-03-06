##  MULTI-LEVEL NETWORK BASED ON TRANSFORMER ENCODER FOR FINE-GRAINED IMAGE-TEXT MATCHING

## 1. Environment

This is a demo on the datasets of MSCOCO and Flickr30k for our paper. We finish experiments on a server with one NVIDIA GeForce 1080Ti GPU.

We recommended the following dependencies.
```
- python 3.6.5
- pyTorch 1.3.1
- numPy (>1.16.1)
- tensorBoard 2.5.0
- nltk 3.6.2
- transformers 2.3.0 
- pytorch-warmup 0.0.4
```

## 2. Relevant data

Please preprocess dataset to appropriate the input format or you can download the data we preprocessed from the pan.baidu.com.
```bash
# f30k
link：https://pan.baidu.com/s/1sjPM1iRXAo30TM4gWQjtSQ 
password：47zx
# coco
link: https://pan.baidu.com/s/1zLEmWuVZItQaZ67FaikFug
password: w1an
```

## 3. Training new models

 Please modify the parameters in the `config.py` to suit your situation.
 #### A. Training global level subnetwork

Run `train-global.py`

#### B.  Training local&relation level subnetwork

Run `train-relation.py`

#### C.  Training digital level subnetwork

Run `train-digital.py`

## 4. Evaluate trained models

Please modify the parameter of `opt.checkpoint` before you run the corresponding file.

#### A. Evaluate global level subnetwork

Run `test-global.py`

#### B.  Evaluate local&relation level subnetwork

Run `test-relation.py`

#### C.  Evaluate digital level subnetwork

Run `test-digital.py`

#### D. Fuse three subnetworks
After you evaluate three subnetworks, you will get three files end of `.mat`.
```
- sims_global.mat
- sims_relation.mat
- sims_digital.mat
```
You can run the `overall-similarity.py` to fuse the three subnetworks to get the overall similarity.



### 4. Thanks

We should thank to these kind researchers, who unreservedly share their source code and advice to us.

Including but not limited to these:

```
 1. Yuxin Peng, Peking University, P.R. China
 2. Fartash Faghri, University of Toronto, Canada
 3. Nicola Messina, ISTI-CNR, Italy
```


### 5. Contact

If you have any question, don't be hesitate to contact Lei Yang at  [864198062@qq.com](mailto:18990848997@163.com).

