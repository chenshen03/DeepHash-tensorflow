# DeepHash

DeepHash is a lightweight deep learning to hash library that implements state-of-the-art deep hashing/quantization algorithms. We will implement more representative deep hashing models continuously according to our released [deep hashing paper list](https://github.com/caoyue10/DeepHashingBaselines). Specifically, we welcome other researchers to contribute deep hashing models into this toolkit based on our framework. We will announce the contribution in this project.

The implemented models include: 

* DQN: [Deep Quantization Network for Efficient Image Retrieval](http://yue-cao.me/doc/deep-quantization-networks-dqn-aaai16.pdf), Yue Cao, Mingsheng Long, Jianmin Wang, Han Zhu, Qingfu Wen, AAAI Conference on Artificial Intelligence (AAAI), 2016
* DHN: [Deep Hashing Network for Efficient Similarity Retrieval](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-hashing-network-aaai16.pdf), Han Zhu, Mingsheng Long, Jianmin Wang, Yue Cao, AAAI Conference on Artificial Intelligence (AAAI), 2016
* DVSQ: [Deep Visual-Semantic Quantization for Efficient Image Retrieval](http://yue-cao.me/doc/deep-visual-semantic-quantization-cvpr17.pdf), Yue Cao, Mingsheng Long, Jianmin Wang, Shichen Liu, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017 
* DCH: [Deep Cauchy Hashing for Hamming Space Retrieval](http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-cauchy-hashing-cvpr18.pdf), Yue Cao, Mingsheng Long, Bin Liu, Jianmin Wang, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018
* DTQ: [Deep Triplet Quantization](ise.thss.tsinghua.edu.cn/~mlong/doc/deep-triplet-quantization-acmmm18.pdf), Bin Liu, Yue Cao, Mingsheng Long, Jianmin Wang, Jingdong Wang, ACM Multimedia (ACMMM), 2018

Note: DTQ and DCH are updated while DQN, DHN, DVSQ maybe outdated, feel free to touch us if you have any questions. We  welcome others to contribute!

## Requirements

-  Python3: Anaconda is recommended because it already contains a lot of packages: 
```
conda create -n DeepHash python=3.6 anaconda
source activate DeepHash
```
-  Other packages: 
```
conda install -y tensorflow-gpu
conda install -y -c conda-forge opencv
```

To import the pakcages implemented in `./DeepHash`, we need to add the path of `./DeepHash` to environment variables as:

```shell
export PYTHONPATH=/path/to/project/DeepHash/DeepHash:$PYTHONPATH
```

## Data Preparation
In `data/cifar10/train.txt`, we give an example to show how to prepare image training data. In `data/cifar10/test.txt` and `data/cifar10/database.txt`, the list of testing and database images could be processed during predicting procedure. If you want to add other datasets as the input, you need to prepare `train.txt`, `test.txt` and `database.txt` as CIFAR-10 dataset.

What's more, We have put the whole cifar10 dataset including the images and data list in the [release page](https://github.com/thulab/DeepHash/releases/download/v0.1/cifar10.zip). You can directly download it and unzip to data/cifar10 folder.

Make sure the tree of `/path/to/project/data/cifar10` looks like this:

```
.
|-- database.txt
|-- test
|-- test.txt
|-- train
`-- train.txt
```

If you need run on NUSWIDE_81 and COCO, we recommend you to follow https://github.com/thuml/HashNet/tree/master/pytorch#datasets to prepare NUSWIDE_81 and COCO images.

For *DVSQ* model, you also need the *word vector* of the semantic labels. Here we use word2vec model pretrained on GoogleNews Dataset (e.g. https://github.com/mmihaltz/word2vec-GoogleNews-vectors), to extract the word embeddings for the labels of images, e.g. dog, cat and so on.

## Get Started

### Pre-trained model

You should manually download the model file of the Imagenet pre-tained AlexNet from [here](https://github.com/thulab/DeepHash/releases/download/v0.1/reference_pretrain.npy.zip) or from release page and unzip it to `/path/to/project/DeepHash/architecture/pretrained_model`.

Make sure the tree of `/path/to/project/DeepHash/architecture` looks like this:

```
├── __init__.py
├── pretrained_model
       └── reference_pretrain.npy
```

### Training and Testing

The example of `$method` (DCH and DTQ) can be run like:

```shell
cd example/$method/
python train_val_script.py --gpus "0,1" --data-dir $PWD/../../data --"other parameters descirbe in train_val_script.py"
```

For DVSQ, DQN and DHN, please refer to the `train_val.sh` and `train_val_script.py` in the examples folder.

## Citations
If you find *DeepHash* is useful for your research, please consider citing the following papers:

    @InProceedings{cite:AAAI16DQN,
      Author = {Yue Cao and Mingsheng Long and Jianmin Wang and Han Zhu and Qingfu Wen},
      Publisher = {AAAI},
      Title = {Deep Quantization Network for Efficient Image Retrieval},
      Year = {2016}
    }
    
    @InProceedings{cite:AAAI16DHN,
      Author = {Han Zhu and Mingsheng Long and Jianmin Wang and Yue Cao},
      Publisher = {AAAI},
      Title = {Deep Hashing Network for Efficient Similarity Retrieval},
      Year = {2016}
    }
    
    @InProceedings{cite:CVPR17DVSQ,
      Title={Deep visual-semantic quantization for efficient image retrieval},
      Author={Cao, Yue and Long, Mingsheng and Wang, Jianmin and Liu, Shichen},
      Booktitle={CVPR},
      Year={2017}
    }
    
    @InProceedings{cite:CVPR18DCH,
      Title={Deep Cauchy Hashing for Hamming Space Retrieval},
      Author={Cao, Yue and Long, Mingsheng and Bin, Liu and Wang, Jianmin},
      Booktitle={CVPR},
      Year={2018}
    }
    
    @article{liu2018deep,
      title={Deep triplet quantization},
      author={Liu, Bin and Cao, Yue and Long, Mingsheng and Wang, Jianmin and Wang, Jingdong},
      journal={MM, ACM},
      year={2018}
    }

## Contacts
Maintainers of  this library:
* Yue Cao, Email: caoyue10@gmail.com
* Bin Liu, Email: liubinthss@gmail.com
