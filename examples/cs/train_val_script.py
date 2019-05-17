import os
import sys
import argparse
import warnings
import data_provider.image as dataset
import model.cs.dqn as model


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_dataset = 'cifar10'
data_root = os.path.join('../../data', _dataset)
label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_81': 81}


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train and val')

    # dataset config
    dataset_group = parser.add_argument_group(title='Dataset config')
    dataset_group.add_argument('--dataset', type=str, default=_dataset)
    dataset_group.add_argument('--data-root', type=str, default=data_root)
    dataset_group.add_argument('--save-dir', type=str, default='./models/')
    dataset_group.add_argument('--log-dir', type=str, default='tflog')
    dataset_group.add_argument('--label-dim', type=str, default=label_dims[_dataset])
    dataset_group.add_argument('--img-tr', type=str, default="{}/train.txt".format(data_root))
    dataset_group.add_argument('--img-te', type=str, default="{}/test.txt".format(data_root))
    dataset_group.add_argument('--img-db', type=str, default="{}/database.txt".format(data_root))
    # network config
    network_group = parser.add_argument_group(title='Network config')
    network_group.add_argument('--gpu', type=int, default=0)
    network_group.add_argument('--max-iter', type=int, default=5000)
    network_group.add_argument('--batch-size', type=int, default=256)
    network_group.add_argument('--val-batch-size', type=int, default=100)
    network_group.add_argument('--decay-step', type=int, default=500, help='Epochs after which learning rate decays')
    network_group.add_argument('--learning-rate-decay-factor', type=float, default=0.5, help='Learning rate decay factor')
    network_group.add_argument('--learning-rate', type=float, default=0.002)
    network_group.add_argument('--img-model', type=str, default='alexnet')
    network_group.add_argument('--model-weights', type=str, default='../../DeepHash/architecture/pretrained_model/reference_pretrain.npy')
    network_group.add_argument('--finetune-all',  type=bool, default=True, help='if only finetune last layer')
    # algorithm config
    algorithm_group = parser.add_argument_group(title='Algorithm config')
    algorithm_group.add_argument('--R', type=int, default=54000)
    algorithm_group.add_argument('--output-dim', type=int, default=64)
    algorithm_group.add_argument('--max-iter-update-b', type=int, default=3)
    algorithm_group.add_argument('--max-iter-update-Cb', type=int, default=1)
    algorithm_group.add_argument('--cq-lambda', type=float, default=0.0001)
    algorithm_group.add_argument('--code-batch-size', type=int, default=500)
    algorithm_group.add_argument('--n-subspace', type=int, default=4)
    algorithm_group.add_argument('--n-subcenter', type=int, default=256)

    return parser.parse_args(argv)


def main(config):
    print(config)
    
    train_img = dataset.import_train(data_root, config.img_tr)
    model_weights = model.train(train_img, config)
    config.model_weights = model_weights
    
#     config.model_weights = './models/lr0.002_cq0.0001_ss4_sc128_d64_cifar10.npy'
    query_img, database_img = dataset.import_validation(data_root, config.img_te, config.img_db)
    maps = model.validation(database_img, query_img, config)

    for key in maps:
        print(("{}: {}".format(key, maps[key])))


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))