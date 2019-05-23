import os
import sys
import argparse
import warnings
import data_provider.image as dataset
import model.dhn.dhn as model


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_21': 21,
              'nuswide_81': 81, 'coco': 80, 'imagenet': 100, 'cifar10_zero_shot': 10}
Rs = {'cifar10': 54000, 'nuswide_81': 5000, 'coco': 5000,
      'nuswide_21': 5000, 'imagenet': 5000, 'cifar10_zero_shot': 15000}


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Train and val')

    # algorithm config
    algorithm_group = parser.add_argument_group(title='Algorithm config')
    algorithm_group.add_argument('--output-dim', type=int, default=48)
    algorithm_group.add_argument('--cq-lambda', type=float, default=0.0)
    algorithm_group.add_argument('--alpha', type=float, default=10) # 10.0 / 0.2
    algorithm_group.add_argument('--loss-type', type=str, default='normed_cross_entropy')

    # network config
    network_group = parser.add_argument_group(title='Network config')
    network_group.add_argument('--gpu', type=int, default=0)
    network_group.add_argument('--max-iter', type=int, default=10000)
    network_group.add_argument('--batch-size', type=int, default=256)
    network_group.add_argument('--val-batch-size', type=int, default=100)
    network_group.add_argument('--decay-step', type=int, default=2000)
    network_group.add_argument('--learning-rate-decay-factor', type=float, default=0.5)
    network_group.add_argument('--learning-rate', type=float, default=0.0001)
    network_group.add_argument('--network', type=str, default='alexnet')
    network_group.add_argument('--network-weights', type=str, default='../../DeepHash/architecture/pretrained_model/reference_pretrain.npy')
    network_group.add_argument('--finetune-all',  type=bool, default=True)
    network_group.add_argument('--test', default=False, action='store_true')

    # dataset config
    dataset_group = parser.add_argument_group(title='Dataset config')
    dataset_group.add_argument('--dataset', type=str, default='cifar10')
    dataset_group.add_argument('--prefix', type=str, default='dhn')
    # config process
    args, rest = parser.parse_known_args()
    _dataset = args.dataset
    _data_root = os.path.join('../../data', _dataset)
    _save_dir = f'snapshot/{args.dataset}_{args.network}_{args.output_dim}bit_{args.prefix}'
    _filename = f'lr{args.learning_rate}_lambda{args.cq_lambda}_alpha{args.alpha}'

    dataset_group.add_argument('--R', type=int, default=Rs[_dataset])
    dataset_group.add_argument('--label-dim', type=str, default=label_dims[_dataset])
    dataset_group.add_argument('--data-root', type=str, default=_data_root)
    dataset_group.add_argument('--img-tr', type=str, default="{}/train.txt".format(_data_root))
    dataset_group.add_argument('--img-te', type=str, default="{}/test.txt".format(_data_root))
    dataset_group.add_argument('--img-db', type=str, default="{}/database.txt".format(_data_root))
    dataset_group.add_argument('--save-dir', type=str, default=_save_dir)
    dataset_group.add_argument('--filename', type=str, default=_filename)

    return parser.parse_args(argv)


def main(config):
    print(config)
    
    if config.test == True:
          config.network_weights = os.path.join(config.save_dir, config.filename+'.npy')
    else:
          train_img = dataset.import_train(config.data_root, config.img_tr)
          network_weights = model.train(train_img, config)
          config.network_weights = network_weights

    query_img, database_img = dataset.import_validation(config.data_root, config.img_te, config.img_db)
    maps = model.validation(database_img, query_img, config)

    for key in maps:
          print(f"{key}: {maps[key]}")

if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))