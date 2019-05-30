import os
import sys
import argparse
import warnings
import data_provider.image as dataset
import model.dqn as model
from util import Logger, str2bool


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

label_dims = {'cifar10': 10, 'cifar10-s1': 10, 'cub': 200, 'nuswide_21': 21,
              'nuswide_81': 81, 'coco': 80, 'imagenet': 100, 'cifar10_zero_shot': 10}

Rs = {'cifar10': 54000, 'cifar10-s1': 50000, 'nuswide_81': 5000, 'coco': 5000,
      'nuswide_21': 5000, 'imagenet': 5000, 'cifar10_zero_shot': 15000}


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train and val')

    # algorithm config
    algorithm_group = parser.add_argument_group(title='Algorithm config')
    algorithm_group.add_argument('--output-dim', type=int, default=64)
    algorithm_group.add_argument('--max-iter-update-b', type=int, default=3)
    algorithm_group.add_argument('--max-iter-update-Cb', type=int, default=1)
    algorithm_group.add_argument('--cq-lambda', type=float, default=0.0001)
    algorithm_group.add_argument('--code-batch-size', type=int, default=500)
    algorithm_group.add_argument('--n-subspace', type=int, default=4)
    algorithm_group.add_argument('--n-subcenter', type=int, default=256)
    # network config
    network_group = parser.add_argument_group(title='Network config')
    network_group.add_argument('--gpu_id', type=str, default='0')
    network_group.add_argument('--max-iter', type=int, default=5000)
    network_group.add_argument('--batch-size', type=int, default=256)
    network_group.add_argument('--val-batch-size', type=int, default=100)
    network_group.add_argument('--decay-step', type=int, default=1000, help='Epochs after which learning rate decays')
    network_group.add_argument('--learning-rate', type=float, default=0.002) # 0.02 for DVSQ, 0.002 for DQN
    network_group.add_argument('--learning-rate-decay-factor', type=float, default=0.5, help='Learning rate decay factor')
    network_group.add_argument('--network', type=str, default='alexnet')
    network_group.add_argument('--network-weights', type=str)
    network_group.add_argument('--finetune-all', type=str2bool, default=True)
    network_group.add_argument('--test', default=False, action='store_true')
    network_group.add_argument('--debug', default=False, action='store_true')
    # dataset config
    dataset_group = parser.add_argument_group(title='Dataset config')
    dataset_group.add_argument('--dataset', type=str, default='cifar10')
    dataset_group.add_argument('--prefix', type=str, default='1')
    # config process
    config, rest = parser.parse_known_args()
    _dataset = config.dataset
    _save_dir = f'../snapshot/{config.dataset}_{config.network}_{config.output_dim}bit_dqn/' + \
            f'{config.prefix}_subspace{config.n_subspace}_subcenter{config.n_subcenter}'
    dataset_group.add_argument('--R', type=int, default=Rs[_dataset])
    dataset_group.add_argument('--label-dim', type=str, default=label_dims[_dataset])
    dataset_group.add_argument('--save-dir', type=str, default=_save_dir)

    return parser.parse_args(argv)


def main(config):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    sys.stdout = Logger(os.path.join(config.save_dir, 'train.log'))

    print(config)
    data_root = os.path.join('../../data', config.dataset)
    img_tr = f'{data_root}/train.txt'
    img_te = f'{data_root}/test.txt'
    img_db = f'{data_root}/database.txt'

    if config.test == True:
        # config.network_weights = os.path.join(config.save_dir, 'network_weights.npy')
        config.network_weights = './models/lr0.002_cq0.0001_ss4_sc256_d64_cifar10.npy'
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