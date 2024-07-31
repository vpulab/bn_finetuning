# VPU Readme

## Installation

### MMSelfSup

Please refer to [install.md](docs/en/install.md) for installation and [prepare_data.md](docs/en/prepare_data.md) for dataset preparation.

[//]: # (- Use Python==3.8, PyTorch==1.7.1, torchvision==0.9.1)

If that doesn't work, try this:

    conda create -n openmmlab python=3.7 -y
    conda activate openmmlab
    conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch -y
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7/index.html
    pip install mmcls

Change directory to the cloned repository (this one, not the official one): 
    
    cd mmselfsup

Finally, run:

    pip install -v -e .


When installing mmselfsup following [install.md](docs/en/install.md), don't clone the official mmselfsup repository at step 3.
Instead use this one.


### Preparing the ISIC-2019 dataset

1. Download the Task 1 training data and ground truth for the ISIC-2019 dataset from: https://challenge.isic-archive.com/data/
2. Unpack the images to your preferred location (for example, datasets/isic2019/images).
3. Create a directory datasets/isic2019/images_color_constancy
4. Apply color correction: `python tools/apply_color_constancy.py` (specifying the path to your ISIC locaiton)
5. Copy the train/test splits to the ISIC directory from mmselfsup/ISIC_2019_test_* and mmselfsup/ISIC_2019_*


## Official tutorials from MMSelfSup

These tutorials provide more details:
- [getting_started.md](docs/en/getting_started.md)
- [config](docs/en/tutorials/0_config.md)
- [add new dataset](docs/en/tutorials/1_new_dataset.md)
- [data pipeline](docs/en/tutorials/2_data_pipeline.md)
- [add new module](docs/en/tutorials/3_new_module.md)
- [customize schedules](docs/en/tutorials/4_schedule.md)
- [customize runtime](docs/en/tutorials/5_runtime.md)
- [benchmarks](docs/en/tutorials/6_benchmarks.md)

## Framework structure

### MMCV

The MMSelfSup framework is based on [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision,
that implements *Runner* objects (see mmcv.runner.base_runner.py for more information) that encapsulate
the model, data loaders, optimizers and loggers. Once initialized, the *Runner* (in
our case *EpochBasedRunner*: mmcv.runner.epoch_based_runner.py) runs the training loop until *max_epochs* is reached.

The training loop has the following stages:

- before_run
- before_train_epoch
- before_train_iter
- after_train_iter
- after_train_epoch
- before_val_epoch
- before_val_iter
- after_val_iter
- after_val_epoch
- after_run

Any stage can be assigned a number of *hooks* (see mmcv.runner.hooks) that include learning rate updaters, evaluation 
(validation), loggers and others. If a hook is registered in one of the stages listed above, it is called whenever
this stage is reached. If many hooks are registered for the same stage their order is defined by their priorities (which
can be changed manually, or left at the default values).

### MMSelfSup

MMSelfSup uses MMCV functionalities to work with Self-Supervised Learning (SSL) models. Specifically, it provides
a versatile architecture that simplifies the initialization and control of *mmcv.runner.** objects. The following
elements are handled by MMSelfSup:

1. Datasets (**mmselfsup.datasets**) defines the dataset logic.
   1. **mmselfsup.datasets.data_sources** describe the dataset representation (i.e. defines how to read annotation files) for different datasets.
   2. **mmselfsup.datasets.single_view** implements ` __get_item__(self, idx)` that returns a *single* view of an image (i.e. an image and its label) - used for classification tasks.
   3. **mmselfsup.datasets.multi_view** implements ` __get_item__(self, idx)` that returns *multiple* views of an image (i.e. *n* augmentations of the original image and its label) - used for some SSL tasks (i.e., contrastive SSL models).
2. Models (**mmselfsup.models**) defines the model logic: backbone, neck, head and algorithm to use.
   1. **mmselfsup.model.backbones** implements two backbone models: ResNet and ResNeXT.
   2. **mmselfsup.models.necks** implements different types of necks to be used (can be omitted) between the head and the backbone.
   3. **mmselfsup.models.heads** implements different types of heads for different use cases.
   4. **mmselfsup.models.memories** implements memory banks for SSL some models.
   5. **mmselfsup.models.algorithms** implements the main logic of the models (SSL or standard classification).
3. Optimizers (**mmselfsup.core.optimizer.optimizers.py**) defines LARS optimizer (extending the list of optimizers in MMCV).
4. Hooks (**mmselfsup.core.hooks**) extends the MMCV's list of hooks by introducing some SSL-related hooks.


## Training an SSL model

To train an SSL model one needs to first define the main configuration of the experiment in the **configs/selfsup/your_architecture/your_config.py**. The configuration file
in this location extends the base configurations of the model, dataset, scheduler and default runtime. You can override some base configurations
in the main configuration file. For example, if your base model config defines a ResNet50 model with 1000 output neurons in the head but
you need only 8, you can add this line in the main configuration file:

    model = dict(head=dict(num_classes=8))

This will override the default value of *num_classes* in the base model config. Similarly, you can override all other settings or add new ones. However, 
if you add many changes (or if they will be reused often) it makes sense to create a new base config file (for example, if you will be using a different dataset).

To start the training of the model use the following command (not that this starts a distributed training on 1 GPU, the reason to use distributed training is that
some models don't support non-distributed training):

    python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29500 tools/train.py configs/.../your_config.py --work_dir work_dirs/.../your_working_directory --seed 0 --launcher pytorch

If you want to initialize the weights of the model using a previously trained model, add this to your command `--cfg-options load_from=your_weights.pth`. Note that the names of the modules (layers) in your 
checkpoint must be prefixed with backbone, neck or head to be loaded correctly. For example, if you try to load the weights with the name *layer4.1.bn2.weight* in your MMSelfSup ResNet50 model you will
get a warning that an unexpected module name was found and it will not be loaded. However, *backbone.layer4.1.bn2.weight* should load successfully. You can check the log files to see if any weights could not
be loaded.

Note that you can override some configuration settings directly from the command line by passing them as cfg-options. For example, you can override the configured value of the starting learning rate
by passing the following to the training call:

    --cfg-options optimizer.lr=0.5


## Training a classification model (fine-tuning)

To train a classification model define the configuration of your experiment in **configs/benchmarks/classification/...** and use the same training call as you do for an SSL model (note
that in this case you don't have to use the distributed training option, but if you do it's okay, the difference will be that you can't debug with PyCharm if you use distributed training).


## Training SSL on ISIC-2019 and transferring to the classification task

Refer to the shell script `train_SSL_with_finetuning_ISIC2019.sh` for the list of commands that run SSL training on ISIC2019 and then
use the trained weights as a starting point for the training of the classifier.

## Adding your own dataset

Refer to the original documentation in MMSelfSup on this topic. Often you don't need to create a new data_source in **mmselfsup.datasets.data_sources**,
you can juts use ImageList if your data annotations look like this (no one-hot encoding): 
 
    img1_name.jpg label
    img2_name.jpg label
    img3_name.jpg label
    ...

Don't forget to use normalization statistics (mean, std) corresponding to your dataset in the dataset configuration file (configs/...). 

## Tips for model training

1. Make sure that `dict(type='TensorboardLoggerHook')` is in the hooks of the log_config in **configs/.../_base_/default_runtime.py**. This will enable tensorboard logging. Use tensorboard logging.
2. By default, tensorboard logs only training loss, training accuracy and whatever is returned by your dataset's **evaluate** method (for example, mmselfsup.datasets.single_view.evaluate). If you want to log a different metric, for example average per-class accuracy, define it in the evaluate method and return it in the **eval_res** dictionary.
3. If you get an assert error from the collect function in the validation stage, make sure you don't have `drop_last=True` anywhere in the dataset configuration. This would tell PyTorch to drop the last batch if it is incomplete (i.e. the size of the dataset isn't divisible by the batch size), which results in the mismatch between the expected length of the dataset and the number of actually loaded samples.
4. You may need to adjust certain SSL configuration settings depending on the dataset you use. For example, the number of classes in ODC (try 80 for ISIC-2019) or the queue length in SwAV (try 256 for ISIC-2019).
5. When you use ODC make sure to change the length of the ODC memory in **configs/selfsup/_base_/models/odc.py** to the number of training instances you have in your dataset. 
