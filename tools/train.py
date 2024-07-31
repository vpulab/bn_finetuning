# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import wandb
import argparse
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmselfsup import __version__
from mmselfsup.apis import init_random_seed, set_random_seed, train_model
from mmselfsup.datasets import build_dataset
from mmselfsup.models import build_algorithm
from mmselfsup.utils import collect_env, get_root_logger
from mmselfsup.datasets.samplers import BalancedSampler, LossSampler, DynamicRandomSampler, DynamicBalancedSampler

import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu_ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--wandb_project',
        type=str)
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default='')
    parser.add_argument(
        '--cifar',
        type=str,
        default="0")
    parser.add_argument(
        '--split',
        type=str,
        default="0")
    parser.add_argument(
        '--linear',
        type=str,
        default="")
    parser.add_argument(
        '--lr_file',
        type=str,
        default="")
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    run = wandb.init(entity='kirill-phd', project=args.wandb_project)
    if args.wandb_run_name != '':
        wandb.run.name = args.wandb_run_name
    cfg = Config.fromfile(args.config)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    artifact = wandb.Artifact(name=f"hyperparameters_{timestamp}", type="config")
    artifact.add_file(local_path=args.config)
    run.log_artifact(artifact)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        work_type = args.config.split('/')[1]
        cfg.work_dir = osp.join('./work_dirs', work_type,
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        assert cfg.model.type not in [
            'DeepCluster', 'MoCo', 'SimCLR', 'ODC', 'NPID', 'DenseCL'
        ], f'{cfg.model.type} does not support non-dist training.'
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init the logger before other steps
    log_file = osp.join(cfg.work_dir, f'train_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed

    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    if args.linear != "":
        if args.linear == 'True':
            cfg.model.head.mlp = False
        else:
            cfg.model.head.mlp = True

    if args.lr_file != "":
        with open(args.lr_file, 'r') as file:
            lr = float(file.readlines()[0])
        cfg.optimizer.lr = lr

    model = build_algorithm(cfg.model)

    if args.cifar != "0":
        # Enlarge the feature maps before pooling (because the images are small: 32x32)
        model.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.backbone.maxpool = torch.nn.Identity()

    if args.split != "0":
        if "EuroSat" in args.wandb_project:
            cfg.data.train.data_source.ann_file = f'/home/kis/code/datasets/EuroSat/split_{args.split}/train.txt'
            cfg.data.val.data_source.ann_file = f'/home/kis/code/datasets/EuroSat/split_{args.split}/val.txt'
        elif "ISIC" in args.wandb_project:
            cfg.data.train.data_source.ann_file = f'/home/kis/code/datasets/ISIC2019/splits/split_{args.split}/train_labeled.txt'
            cfg.data.val.data_source.ann_file = f'/home/kis/code/datasets/ISIC2019/splits/split_{args.split}/val_labeled.txt'
        elif "Chest" in args.wandb_project:
            cfg.data.train.data_source.ann_file = f'/home/kis/code/datasets/chest_x_ray/split_{args.split}/train.txt'
            cfg.data.val.data_source.ann_file = f'/home/kis/code/datasets/chest_x_ray/split_{args.split}/val.txt'
        elif "DTD" in args.wandb_project:
            cfg.data.train.data_source.ann_file = f'/home/kis/code/datasets/dtd/labels/trainval_{args.split}_labeled.txt'
            cfg.data.val.data_source.ann_file = f'/home/kis/code/datasets/dtd/labels/test_{args.split}_labeled.txt'

    print(model)
    model.init_weights()


    datasets = [build_dataset(cfg.data.train)]
    if "sampler_name" in cfg.data and cfg.data["sampler_name"] == "dynamic_balanced_sampler":
        print(f"Using custom data loader: {cfg.data.sampler_name}")
        with open(cfg.data.train.data_source["ann_file"], "r") as annotations:
            labels = [int(line.strip().split(" ")[1]) for line in annotations.readlines()]
        sampler = DynamicBalancedSampler(labels,
                                             datasets[0],
                                             subset_size=1, # use 100% of the training data
                                             refresh_rate=-1,
                                             num_classes=cfg.data.n_classes,
                                             batch_size=cfg.data.imgs_per_gpu
                                             )
        cfg.data.train.sampler = sampler
    elif "sampler_name" in cfg.data and cfg.data["sampler_name"] == "dynamic_random_sampler":
        print(f"Using custom data loader: {cfg.data.sampler_name}")
        with open(cfg.data.train.data_source["ann_file"], "r") as annotations:
            labels = [int(line.strip().split(" ")[1]) for line in annotations.readlines()]
        sampler = DynamicRandomSampler(labels,
                                             datasets[0],
                                             subset_size=float(cfg.data.subset_size),
                                             refresh_rate=8,
                                             num_classes=cfg.data.n_classes,
                                             batch_size=cfg.data.imgs_per_gpu
                                             )
        cfg.data.train.sampler = sampler
    

    assert len(cfg.workflow) == 1, 'Validation is called by hook.'
    if cfg.checkpoint_config is not None:
        # save mmselfsup version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmselfsup_version=__version__, config=cfg.pretty_text)

    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
