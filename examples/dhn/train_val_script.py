import numpy as np
import scipy.io as sio
import warnings
import data_provider.image as dataset
import model.dhn.dhn as model
import sys
from pprint import pprint
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define input arguments
lr = float(sys.argv[1])
output_dim = int(sys.argv[2])
iter_num = int(sys.argv[3])
cq_lambda = float(sys.argv[4])
alpha = float(sys.argv[5])
_dataset = sys.argv[6]
gpu = sys.argv[7]
log_dir = sys.argv[8]
data_root = sys.argv[9]

label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_21': 21,
              'nuswide_81': 81, 'coco': 80, 'imagenet': 100, 'cifar10_zero_shot': 10}
Rs = {'cifar10': 54000, 'nuswide_81': 5000, 'coco': 5000,
      'nuswide_21': 5000, 'imagenet': 5000, 'cifar10_zero_shot': 15000}

config = {
    'device': '/gpu:' + gpu,
    'max_iter': iter_num,
    'batch_size': 256,  # TODO
    'val_batch_size': 100,
    'decay_step': 5000,  # TODO     # Epochs after which learning rate decays.
    'learning_rate_decay_factor': 0.5,   # Learning rate decay factor.
    'learning_rate': lr,                 # Initial learning rate img.

    'output_dim': output_dim,
    'alpha': alpha,

    'R': Rs[_dataset],
    'model_weights': '../../DeepHash/architecture/pretrained_model/reference_pretrain.npy',

    'img_model': 'alexnet',
    # 'loss_type': 'normed_cross_entropy',  # normed_cross_entropy # TODO
    'loss_type': 'cross_entropy',

    # if only finetune last layer
    'finetune_all': True,

    # CQ params
    'cq_lambda': cq_lambda,

    'label_dim': label_dims[_dataset],
    'img_tr': "{}/train.txt".format(data_root),
    'img_te': "{}/test.txt".format(data_root),
    'img_db': "{}/database.txt".format(data_root),
    'save_dir': "./models/",
    'log_dir': log_dir,
    'dataset': _dataset
}

pprint(config)

train_img = dataset.import_train(data_root, config['img_tr'])
model_weights = model.train(train_img, config)

config['model_weights'] = model_weights
query_img, database_img = dataset.import_validation(data_root, config['img_te'], config['img_db'])
maps = model.validation(database_img, query_img, config)
for key in maps:
    print(("{}: {}".format(key, maps[key])))
pprint(config)
