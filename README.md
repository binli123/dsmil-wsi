# DSMIL: Dual-stream multiple instance learning networks for tumor detection in Whole Slide Image
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
  $ python download.py --dataset=mil
```

Precomputed features for [TCGA Lung Cancer dataset](https://portal.gdc.cancer.gov/repository?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.cases.primary_site%22%2C%22value%22%3A%5B%22bronchus%20and%20lung%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_format%22%2C%22value%22%3A%5B%22svs%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.experimental_strategy%22%2C%22value%22%3A%5B%22Diagnostic%20Slide%22%5D%7D%7D%5D%7D) can be downloaded via:  
```
  $ python download.py --dataset=tcga
```
This dataset requires 20GB of free disk space.

## **Training on default datasets.**  
Train DSMIL on standard MIL benchmark dataset:
```
  $ python train_mil.py
```
Switch between MIL benchmark dataset, use option:
 ```
 [--datasets]      # musk1, musk2, elephant, fox, tiger
 ```
Other options are available for learning rate (0.0002), cross validation fold (5), weight-decay (5e-3), and number of epochs (40).  
 
Train DSMIL on TCGA Lung Cancer dataset (precomputed features):
 ```
  $ python train_tcga.py --simclr=0
```

## Process WSI data
If you are processing WSI from raw images, you will need to download the WSIs first.  
1. **Download WSIs.**  
Navigate to './tcga-download/' and download WSIs from [TCGA data portal](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Getting_Started/) using the manifest file and configuration file.  
The example shows the case of Windows operating system. The WSIs will be saved in './WSI/TCGA-lung/LUAD' and './WSI/TCGA-lung/LUSC'.  
The raw WSIs take about 1TB disc space and may take several days to download. Open command line tool (*Command Prompt* for the case of Windows), navigate to './tcga-download', and use commands:
```
  $ cd tcga-download
  $ gdc-client -m gdc_manifest.2020-09-06-TCGA-LUAD.txt --config config-LUAD.dtt
  $ gdc-client -m gdc_manifest.2020-09-06-TCGA-LUSC.txt --config config-LUSC.dtt
```    
2. **Prepare the patches.**  
We will be using [OpenSlide](https://openslide.org/), a C library with a [Python API](https://pypi.org/project/openslide-python/) that provides a simple interface to read WSI data. We refer the users to [OpenSlide Python API document](https://openslide.org/api/python/) for the details of using this tool.  
The patches will be saved in './WSI/TCGA-lung/pyramid' in a pyramidal structure for the magnifications of 20x and 5x. Navigate to './tcga-download/OpenSlide/bin' and run the script 'TCGA-pre-crop.py'  
```
  cd tcga-download/OpenSlide/bin
  $ python TCGA-pre-crop.py
```
3. **Train the embedder.**  
We provided a modified script from this repository [Pytorch implementation of SimCLR](https://github.com/sthalles/SimCLR) For training the embedder.  
Navigate to './simclr' and edit the attributes in the configuration file 'config.yaml'. You will need to determine a batch size that fits your gpu(s). We recommand to use a batch size of at least 512 to get good simclr features. The trained model weights and loss log are saved in folder './simclr/runs'.
```
  cd simclr
  $ python run.py
```
4. **Compute the features.**  
```
  $ python compute_feats.py
```
5. **Start training.**  
```
  $ python train_tcga.py --simclr=1
```

## Training on your own datasets
You could modify train_tcga.py to easily let it work with your datasets. After you have trained your embedder, you will need to compute the features and organize them as:  
1. For each bag, generate a .csv file where each row contains the feature of an instance. The .csv file should be named as "_bagID_.csv" and put into a folder named "_dataset-name_".  

<div align="center">
  <img src="thumbnails/bag.png" width="700px" />
</div>  

2. Generate a "_dataset-name_.csv" file with two columns where the first column contains the paths to all _bagID_.csv files, and the second column contains the bag labels.  

<div align="center">
  <img src="thumbnails/bags.png" width="700px" />
</div>  

3. Replace the corresponding file path in the script with the file path of "_dataset_.csv".  
```
  bags_path = pd.read_csv(PATH_TO_[dataset-name.csv])
```
4. Configure the corresponding number of classes argument for creating the DSMIL model.
5. Start training.  
```
  $ python train_tcga.py --simclr=1
```

## Citation
If you use the code or results in your research, please use the following BibTeX entry.  
```
@article{li2020dualstream,
  author =   {Bin Li and Yin Li and Kevin W. Eliceiri},
  title =    {Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification with Self-supervised Contrastive Learning},
  journal =  {arXiv preprint arXiv:2011.08939},
  year =     {2020}
}


