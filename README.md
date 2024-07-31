# Improved Transferability of Self-Supervised Learning Models Through Batch Normalization Finetuning

Abundance of unlabelled data and advances in Self-Supervised Learning (SSL) have made it the preferred choice in many transfer learning scenarios. Due to the rapid and ongoing development of SSL approaches, practitioners are now
faced with an overwhelming amount of models trained for a specific task/domain, calling for a method to estimate transfer performance on novel tasks/domains. Typically, the role of such estimator is played by linear probing which trains a linear classifier on top of the frozen feature extractor. In this work we address a shortcoming of linear probing —it is not very strongly correlated with the performance of the models finetuned end-to-end—the latter often being the final objective in transfer learning—and, in some cases, catastrophically misestimates a model’s potential. We propose a way to obtain a significantly better proxy task by unfreezing and jointly finetuning batch normalization layers together with the classification head. At a cost of extra training of only 0.16% model parameters, in case of ResNet-50, we acquire a proxy task that (i) has a stronger correlation with
end-to-end finetuned performance, (ii) improves the linear probing performance in the many- and few-shot learning regimes and (iii) in some cases, outperforms both linear probing and end-to-end finetuning, reaching the state-of-the-art performance on a pathology dataset. Finally, we analyze and discuss the changes batch normalization training introduces in the feature distributions that may be the reason for the improved performance.

![Linear probing vs BN-tuning](https://github.com/user-attachments/assets/a74c53af-8dfb-418c-a622-e0032e9fd89a)

## Experimental setup

In this work we demonstrate the benefits of finetuning BN affines during SSL linear probing in many- and few-shot regimes. Specifically, in the many-shot setup we train 12 SSL models and compare obtained results to standard linear probing and end-to-end finetuning. We use few-shot learning benchmark datasets to further show that BN finetuning is advantageous for SSL model evaluation in scenarios with limited training data and strong domain shifts. 

To replicate the experiments in the paper you will need to install the environment, download the data and the pretrained models.

### Environment

This project is based on the mmselfsup framework. Please refer to [install.md](docs/en/install.md) for installation and [prepare_data.md](docs/en/prepare_data.md) for dataset preparation.

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


When installing mmselfsup following [install.md](docs/en/install.md), don't clone the official mmselfsup repository at step 3. Instead use this one.

Then install the remaining missing packages from environment.yml.

### Datasets
Download the following datasets and place them into the ./data directory of the project. The following structure is expected: 

    ./data/DATASET_NAME/IMG_FOLDER_NAME/IMG_0_NAME.***  
    
    ./data/DATASET_NAME/IMG_FOLDER_NAME/IMG_1_NAME.***  
    
    ./data/DATASET_NAME/IMG_FOLDER_NAME/IMG_2_NAME.***  
    
    ...


The annotations files are expected to have the following format:



    IMG_0_NAME CLASS_NUMBER  
    
    IMG_1_NAME CLASS_NUMBER
    
    IMG_2_NAME CLASS_NUMBER
    
    ...



- DTD (https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- Caltech-101 (https://data.caltech.edu/records/mzrjq-6wc02)
- FGVC Aircraft (https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- Stanford cars (https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset - the original URL is broken)
- Oxford-IIIT Pets (https://www.robots.ox.ac.uk/~vgg/data/pets/)
- Oxford 102 Flowers (https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- CIFAR-10 (https://www.cs.toronto.edu/~kriz/cifar.html)
- CIFAR-100 (https://www.cs.toronto.edu/~kriz/cifar.html)

- EuroSAT (https://zenodo.org/records/7711810#.ZAm3k-zMKEA)
- NIH Chest X-Ray (https://nihcc.app.box.com/v/ChestXray-NIHCC)
- ISIC-2018 (https://challenge.isic-archive.com/data/#2018)
- MHIST (https://bmirds.github.io/MHIST/) - requires registration to access the data

### Models

Download the pretrained models from **xxxxxxxx**  and place them in ./pretrained_models/official_weights/mmselfsup_format

### Train the models

To transfer an SSL model with **BN-finetuning** to a downstream dataset (for example, SwAV model to DTD dataset), use the following command:

    python3 tools/train.py ./configs/benchmarks/classification/dtd/resnet50_train_bn.py --work_dir ./work_dirs/dtd_train_bn/swav --cfg-options load_from=./pretrained_models/official_weights/mmselfsup_format/swav_backbone.pth --split 0 --wandb_project "DTD Train BN" --linear True --wandb_run_name "SwAV, split 0"

To transfer an SSL model with **linear probing** to a downstream dataset (for example, SwAV model to DTD dataset), use the following command:

    python3 tools/train.py ./configs/benchmarks/classification/dtd/resnet50.py --work_dir ./work_dirs/dtd_train_linear/swav --cfg-options load_from=./pretrained_models/official_weights/mmselfsup_format/swav_backbone.pth --split 0 --wandb_project "DTD Linear" --linear True --wandb_run_name "SwAV, split 0"

To train the models in the few-shot regime, please refer to https://github.com/linusericsson/ssl-transfer/blob/main/few_shot.py


 
