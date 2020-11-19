  
import urllib.request
import argparse
from tqdm import tqdm
import zipfile
import shutil
import os


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        
def unzip_data(zip_path, data_path):
    if os.path.exists(data_path): shutil.rmtree(data_path) 
    os.mkdir(data_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, 
                        default='MIL', 
                        help='Dataset to be downloaded: MIL, TCGA')
    args = parser.parse_args()
    
    if args.dataset == "MIL":
        print('downloading MIL benchmark datasets')
        download_url('https://docs.google.com/uc?export=download&id=1MvG1zA3IkWSzPFNPczDRG7bGeVULBcY-', 'datasets')
        unzip_data('datasets/mil-dataset.zip', 'datasets')
        os.remove('datasets/mil-dataset.zip')
    if args.dataset == "TCGA":
        print('downloading TCGA Lung Cancer datasets (pre-computed features)')
        download_url('https://docs.google.com/uc?export=download&id=1xfWfnJ53fWczem86lDTjJhO1UTZ-hyXO', 'datasets')
        unzip_data('datasets/tcga-dataset.zip', 'datasets')
        os.remove('datasets/tcga-dataset.zip')
    
if __name__ == '__main__':
    main()