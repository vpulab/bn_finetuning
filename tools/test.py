# Copyright (c) OpenMMLab. All rights reserved.
import ttach as tta
import pickle
import argparse
import os
import os.path as osp
import time

import mmcv
import numpy as np
import torch

from torch.nn import Sigmoid
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmselfsup.datasets import build_dataloader, build_dataset
from mmselfsup.models import build_algorithm
from mmselfsup.utils import get_root_logger, multi_gpu_test, single_gpu_test, single_gpu_test_losses


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
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
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
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

    # create work_dir  
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'test_{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=cfg.data.imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        use_cutmix=False)

    # build the model and load checkpoint
    model = build_algorithm(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')


    outputs = []
    number_of_augmentations = 1



    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        for _ in range(number_of_augmentations):
            outputs.append(single_gpu_test(model, data_loader)['head4'])
        outputs = dict(head4=np.add.reduce(outputs) / len(outputs))
        # outputs = single_gpu_test(model, data_loader)['head4']
        # outputs = single_gpu_test_losses(model, data_loader)['loss']

    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader)  # dict{key: np.ndarray}



    # with open(f"{cfg.work_dir}/test_{args.checkpoint.split('/')[-1].split('.')[0]}_outputs.pkl", "wb") as f:
    #     pickle.dump(outputs, f)



    # np.std(val_df.values.reshape(-1))
    # print(sigmoided_ouputs)
    
    with open(f"{cfg.work_dir}/outputs_{args.work_dir.split('/')[-1]}.txt", "w") as f:
        for pred in outputs['head4']:
            f.write(f"{pred}\n")
    
    s = Sigmoid()
    sigmoided_ouputs = s(torch.tensor(outputs['head4']))
    
    with open(f"{cfg.work_dir}/outputs__{args.work_dir.split('/')[-1]}_sigmoided.txt", "w") as f:
        for pred in sigmoided_ouputs:
            f.write(f"{pred}\n")

    with open(f"{cfg.work_dir}/predictions__{args.work_dir.split('/')[-1]}.txt", "w") as f:
        for pred in sigmoided_ouputs:
            f.write(f"{torch.argmax(pred)}\n")
        
    # rank, _ = get_dist_info()
    # if rank == 0:
    # evaluation = dict(interval=1, start=0, save_best="auto", topk= (1,), per_class=True, n_classes= n_classes)

    dataset.evaluate(outputs, logger, topk=(1, ), per_class=False, n_classes=52)
    


if __name__ == '__main__':
    main()
