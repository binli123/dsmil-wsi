# DSMIL: Dual-stream multiple instance learning networks for tumor detection in Whole Slide Image
This is the Pytorch implementation for the multiple instance learning model described in the paper [Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning](https://arxiv.org/abs/2011.08939) (_CVPR 2021, accepted for oral presentation_).  

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
Install [OpenSlide and openslide-python](https://pypi.org/project/openslide-python/).  
[Tutorial 1](https://openslide.org/) and [Tutorial 2 (Windows)](https://www.youtube.com/watch?v=0i75hfLlPsw).  

## Features preparation
MIL benchmark datasets can be downloaded via:
```
  $ python download.py --dataset=mil
```

Precomputed features for [TCGA Lung Cancer dataset](https://portal.gdc.cancer.gov/repository?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.cases.primary_site%22%2C%22value%22%3A%5B%22bronchus%20and%20lung%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_format%22%2C%22value%22%3A%5B%22svs%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22Diagnostic%20Slide%22%5D%7D%7D%5D%7D) can be downloaded via:  
```
  $ python download.py --dataset=tcga
```
This dataset requires 20GB of free disk space.

## Training on default datasets.
### MIL benchmark datasets
>Train DSMIL on standard MIL benchmark dataset:
```
  $ python train_mil.py
```
>Switch between MIL benchmark dataset, use option:
 ```
 [--datasets]      # musk1, musk2, elephant, fox, tiger
 ```
>Other options are available for learning rate (--lr=0.0002), cross validation fold (--cv_fold=10), weight-decay (--weight_decay=5e-3), and number of epochs (--num_epoch=40).  
### TCGA lung datasets
>Train DSMIL on TCGA Lung Cancer dataset (precomputed features):
 ```
  $ python train_tcga.py --new_features=0
```
## Testing on TCGA lung dataset
>We provided a testing pipeline for several sample slides. The slides can be downloaded via:  
```
  $ python download.py --dataset=tcga-test
```   
>To crop the WSIs into patches, run:  
```
  $ python TCGA_test_10x.py
```  
>A folder containing all patches for each WSI will be created at `./test/patches`.  
>After the WSIs are cropped, run the testing script:
```
  $ python testing.py
```   
>The thumbnails of the WSIs will be saved in `./test/thumbnails`.  
>The detection color maps will be saved in `./test/output`.  
>The testing pipeline will process every WSI placed inside the `./test/input` folder. The slide will be detected as a LUAD, LUSC, or benign sample.   

## Processing raw WSI data
If you are processing WSI from raw images, you will need to download the WSIs first.

**Download WSIs.**  
>**From GDC data portal.** You can use [GDC data portal](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/) with a manifest file and configuration file. Example manifest files and configuration files can be found in tcga-download. 
The raw WSIs take about 1TB of disc space and may take several days to download. Open command-line tool (*Command Prompt* for the case of Windows), unzip the data portal client into `./tcga-download`, navigate to `./tcga-download`, and use commands:
```
  $ cd tcga-download
  $ gdc-client -m gdc_manifest.2020-09-06-TCGA-LUAD.txt --config config-LUAD.dtt
  $ gdc-client -m gdc_manifest.2020-09-06-TCGA-LUSC.txt --config config-LUSC.dtt
```
>The data will be saved in `./WSI/TCGA-lung`. Please check [details](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/) regarding the use of TCGA data portal. Otherwise, individual WSIs can be download manually in GDC data portal [repository](https://portal.gdc.cancer.gov/repository?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22content%22%3A%7B%22field%22%3A%22files.cases.primary_site%22%2C%22value%22%3A%5B%22bronchus%20and%20lung%22%5D%7D%2C%22op%22%3A%22in%22%7D%2C%7B%22content%22%3A%7B%22field%22%3A%22files.data_format%22%2C%22value%22%3A%5B%22svs%22%5D%7D%2C%22op%22%3A%22in%22%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22Diagnostic%20Slide%22%5D%7D%7D%5D%7D)  
>**From Google Drive.** The svs files are also [uploaded](https://drive.google.com/drive/folders/1UobMSqJEqINX2izxrwbgprugjlTporSQ?usp=sharing). The dataset contains in total 1053 slides, including 512 LUSC and 541 LUAD. 10 low-quality LUAD slides are discarded. 

**Prepare the patches.**  
>We will be using [OpenSlide](https://openslide.org/), a C library with a [Python API](https://pypi.org/project/openslide-python/) that provides a simple interface to read WSI data. We refer the users to [OpenSlide Python API document](https://openslide.org/api/python/) for the details of using this tool.  
>The patches could be saved in './WSI/TCGA-lung/pyramid' in a pyramidal structure for the magnifications of 20x and 5x. Run:  
```
  $ python WSI_cropping.py --multiscale=1
```
>Or, the patches could be cropped at a single magnification of 10x and saved in './WSI/TCGA-lung/single' by using:  
```
  $ python WSI_cropping.py --multiscale=0
```

**Train the embedder.**  
>We provided a modified script from this repository [Pytorch implementation of SimCLR](https://github.com/sthalles/SimCLR) For training the embedder.  
Navigate to './simclr' and edit the attributes in the configuration file 'config.yaml'. You will need to determine a batch size that fits your gpu(s). We recommend using a batch size of at least 512 to get good simclr features. The trained model weights and loss log are saved in folder './simclr/runs'.
```
  cd simclr
  $ python run.py
```

**Compute the features.**  
>Compute the features for 20x magnification:  
```
  $ cd ..
  $ python compute_feats.py --dataset=wsi-tcga-lung
```
>Or, compute the features for 10x magnification:  
```
  $ python compute_feats.py --dataset=wsi-tcga-lung-single --magnification=10x
```

**Start training.**  
```
  $ python train_tcga.py --new_features=1
```

## Training on your own datasets
1. Place WSI files into `WSI\[DATASET_NAME]\[CATEGORY_NAME]\[SLIDE_FOLDER_NAME] (optional)\SLIDE_NAME.svs`. 
> For binary classifier, the negative class should have `[CATEGORY_NAME]` at index `0` when sorted alphabetically. For multi-class classifier, if you have a negative class (not belonging to any of the positive classes), the folder should have `[CATEGORY_NAME]` at the last index when sorted alphabetically. The naming does not matter if you do not have a negative class.
2. Crop patches.  
```
  $ python WSI_cropping.py --dataset=[DATASET_NAME]
```
3. Train an embedder.  
```
  $ cd simclr
  $ python run.py --dataset=[DATASET_NAME]
```
4. Compute features using the embedder.  
```
  $ cd ..
  $ python compute_feats.py --dataset=[DATASET_NAME]
```
>This will use the last trained embedder to compute the features, if you want to use an embedder from a specific run, add the option `--weights=[RUN_NAME]`, where `[RUN_NAME]` is a folder name inside `simclr/runs/`. If you have an embedder you want to use, you can place the weight file as `simclr/runs/[FOLDER_NAME]/checkpoints/model.pth` and pass the `[FOLDER_NAME]` to this option. The embedder architecture is ResNet18.    
5. Training.
```
  $ python train_tcga.py --dataset=[DATASET_NAME] --new_features=1
```
>You will need to adjust `--num_classes` option if the dataset contains more than 2 positive classes or only 1 positive class. See the next section for details.  
  
## Feature vector csv files explanation
1. For each bag, generate a .csv file where each row contains the feature of an instance. The .csv file should be named as "_bagID_.csv" and put into a folder named "_dataset-name_/_category_/".  

<div align="center">
  <img src="thumbnails/bag.png" width="700px" />
</div>  

2. Generate a "_dataset-name_.csv" file with two columns where the first column contains the paths to all _bagID_.csv files, and the second column contains the bag labels.  

<div align="center">
  <img src="thumbnails/bags.png" width="700px" />
</div>  

3. Labels.
> For binary classifier, use `1` for positive bags and `0` for negative bags. Use `--num_classes=1` at training.  
> For multi-class classifier (`N` positive classes and one optional negative class), use `0~(N-1)` for positive classes. If you have negative class (not belonging to any one of the positive classes), use `N` for its label. Use `--num_classes=N` (`N` equals the number of **positive** classes) at training.


## Citation
If you use the code or results in your research, please use the following BibTeX entry.  
```
@article{li2020dual,
  title={Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning},
  author={Li, Bin and Li, Yin and Eliceiri, Kevin W},
  journal={arXiv preprint arXiv:2011.08939},
  year={2020}
}


