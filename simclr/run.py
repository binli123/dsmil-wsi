from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import os, glob
import pandas as pd
import argparse

def generate_csv(level='20x'):
    if level=='20x':
        patch_path = glob.glob('../WSI/TCGA-lung/pyramid/*/*/*/*.jpg') # /class_name/bag_name/5x_name/*.jpg
    else:
        patch_path = glob.glob('../WSI/TCGA-lung/pyramid/*/*/*.jpg') # /class_name/bag_name/*.jpg
    df = pd.DataFrame(patch_path)
    df.to_csv('all_patches.csv', index=False)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--magnification', type=str, default='20x')
    args = parser.parse_args()
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])
    
    generate_csv(args.magnification)
    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
