# DSMIL: Multiple instance learning networks for tumor detection in Whole Slide Image
This is the Pytorch implementation for the multiple instance learning model described in the paper [Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning](https://arxiv.org/abs/2011.08939). 

<div align="center">
  <img src="thumbnails/overview.png" width="700px" />
</div>

<div align="center">
  <img src="thumbnails/overview-2.png" width="700px" />
</div>

## Installation
Install [anaconda/miniconda](https://docs.conda.io/en/latest/miniconda.html)  
Required packages
```
  $ conda env create --name dsmil --file env.yml
  $ conda activate dsmil
```

## Features preparation
The MIL benchmark dataset can be downloaded via:
```
  $ python download.py --dataset=MIL
```

If you are processing WSI data, you will need a pretrained embedder and precompute the features of each patch.  
* Your WSIs must be cropped into patches first. [OpenSlide](https://openslide.org/) is a C library with a [Python API](https://pypi.org/project/openslide-python/) that provides a simple interface to read WSI data. We refer the users to [OpenSlide Python API document](https://openslide.org/api/python/) for the details of using this tool.    
* For training your embedder, we refer the users to [Pytorch implementation of SimCLR](https://github.com/sthalles/SimCLR). You would need to feed your WSI patches to the SimCLR framework with "input_shape" argument set as the size of the WSI patch in the configuration file (config.yaml).  

Otherwise, precomputed features for [TCGA Lung Cancer dataset](https://portal.gdc.cancer.gov/repository?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.cases.primary_site%22%2C%22value%22%3A%5B%22bronchus%20and%20lung%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_format%22%2C%22value%22%3A%5B%22svs%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22Diagnostic%20Slide%22%5D%7D%7D%5D%7D) can be downloaded via:  
```
  $ python download.py --dataset=TCGA
```

## Training on default datasets
To train DSMIL on standard MIL benchmark dataset:
```
  $ python train_mil.py
```
To switch between MIL benchmark dataset, use option:
 ```
 [--datasets]      # musk1, musk2, elephant, fox, tiger
 ```
 To train DSMIL on TCGA Lung Cancer dataset:
 ```
  $ python train_tcga.py
```

## Training on your own datasets
You could modify train_tcga.py to easily let it work on your datasets. You will need to:  
1. For each bag, generate a .csv file where each row contains the feature of a instance. The .csv file should be named as "_bagID_.csv" and put into a folder named "_dataset-name_".  
2. Generate a "_dataset-name_.csv" file with two columns where the first column contains _bagID_, and the second column contains the class label.
3. Replace the corresponding file path in the script with the file path of "_dataset_.csv" file, and change the data directory path in the dataloader to the path of the folder "_dataset-name_"


