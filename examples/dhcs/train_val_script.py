import os
import sys
import argparse
import warnings
import data_provider.image as dataset
import model.dhcs as model
from pprint import pprint
from util import Logger, str2bool


label_dims = {'cifar10': 10, 'cifar10-s1': 10, 'cub': 200, 'nuswide_21': 21,
              'nuswide_81': 81, 'coco': 80, 'imagenet': 100, 'cifar10_zero_shot': 10}

Rs = {'cifar10': 54000, 'cifar10-s1': 50000, 'nuswide_81': 5000, 'coco': 5000,
      'nuswide_21': 5000, 'imagenet': 5000, 'cifar10_zero_shot': 15000}


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train and val model')

    # algorithm config
    algorithm_group = parser.add_argument_group(title='Algorithm config')
    algorithm_group.add_argument('--bit', type=int, default=32)
    algorithm_group.add_argument('--q-lambda', type=float, default=0.01)
    algorithm_group.add_argument('--b-lambda', type=float, default=0.0)
    algorithm_group.add_argument('--alpha', type=float, default=10)
    # network config
    network_group = parser.add_argument_group(title='Network config')
    network_group.add_argument('--gpus', type=str, default='0')
    network_group.add_argument('--max-iter', type=int, default=5000)
    network_group.add_argument('--batch-size', type=int, default=128)
    network_group.add_argument('--val-batch-size', type=int, default=100)
    network_group.add_argument('--decay-step', type=int, default=3000) 
    network_group.add_argument('--lr', type=float, default=0.005)
    network_group.add_argument('--learning-rate-decay-factor', type=float, default=0.5)
    network_group.add_argument('--network', type=str, default='alexnet')
    network_group.add_argument('--network-weights', type=str)
    network_group.add_argument('--finetune-all',  type=str2bool, default=True)
    network_group.add_argument('--test', default=False, action='store_true')
    network_group.add_argument('--debug', default=False, action='store_true')
    # dataset config
    dataset_group = parser.add_argument_group(title='Dataset config')
    dataset_group.add_argument('--dataset', type=str, default='cifar10')
    dataset_group.add_argument('--prefix', type=str, default='1')
    # config process
    config, rest = parser.parse_known_args()
    _dataset = config.dataset
    _save_dir = f'../snapshot/{config.dataset}_{config.network}_{config.bit}bit_exp/' + \
        f'{config.prefix}_lr{config.lr}_a{config.alpha}_q{config.q_lambda}_b{config.b_lambda}'
    dataset_group.add_argument('--R', type=int, default=Rs[_dataset])
    dataset_group.add_argument('--label-dim', type=str, default=label_dims[_dataset])
    dataset_group.add_argument('--save-dir', type=str, default=_save_dir)

    return parser.parse_args(argv)


def main(config):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    sys.stdout = Logger(os.path.join(config.save_dir, 'train.log'))

    pprint(vars(config))
    data_root = os.path.join('../../data', config.dataset)
    config.wordvec_dict = f'{data_root}/wordvec.txt'
    img_tr = f'{data_root}/train.txt'
    img_te = f'{data_root}/test.txt'
    img_db = f'{data_root}/database.txt'

    if config.test == True:
        config.network_weights = os.path.join(config.save_dir, 'network_weights.npy')
    else:
        train_img = dataset.import_train(data_root, img_tr)
        network_weights = model.train(train_img, config)
        config.network_weights = network_weights

    query_img, database_img = dataset.import_validation(data_root, img_te, img_db)
    maps = model.validation(database_img, query_img, config)

    for key in maps:
        print(f"{key}: {maps[key]}")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))