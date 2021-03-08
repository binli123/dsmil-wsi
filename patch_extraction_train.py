import subprocess, os, re, time, sys, warnings, glob
import openslide as slide
from PIL import Image
import numpy as np
from skimage import data, io, transform
from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_ubyte, view_as_windows
from skimage import img_as_ubyte
from os import listdir, mkdir, path, makedirs
from os.path import join 
from tqdm import tqdm
import argparse

def thres_saturation(img, t=15):
    # typical t = 15
    img = rgb2hsv(img)
    h, w, c = img.shape
    sat_img = img[:, :, 1]
    sat_img = img_as_ubyte(sat_img)
    ave_sat = np.sum(sat_img) / (h * w)
    return ave_sat >= t
                        
def slide_to_patch(out_base, img_slides, patch_size, overlap, num_worker):
    makedirs(out_base, exist_ok=True)
    for s in tqdm(range(len(img_slides))):
        img_slide = img_slides[s]
        img_name = img_slide.split(path.sep)[-1].split('.')[0]
        img_class = img_slide.split(path.sep)[-3]
        bag_path = join(out_base, img_class, img_name)
        makedirs(bag_path, exist_ok=True)
        img = slide.OpenSlide(img_slide)
        if int(np.floor(float(img.properties['openslide.mpp-x'])*10)) == 2: # 40x scan
            dimension_5x = -4
        else:
            dimension_5x = -3
        subprocess.call(['python', 'deep-zoom-tiler.py', img_slide, '-o', bag_path, '-s', str(patch_size), '-e', str(overlap), '-j', str(num_worker)])
        levels = glob.glob(os.path.join(bag_path, '*'))
        levels.sort(key=lambda f: int(re.sub('\D', '', f)), reverse=True)
        base_mag = 40
        for idx, level in enumerate(levels):
            mag = str(base_mag / 2**(idx)) + 'x'
            mag_name = os.path.join(*level.split(os.sep)[0:-1]) + os.sep + mag
            os.rename(level, mag_name)
        os.remove(bag_path+'.dzi')
        all_patches = glob.glob(os.path.join(bag_path, '*', '*'))
        print('Removing backgrounds')
        for idx, patch in enumerate(all_patches):
            sys.stdout.write('\r {}/{}'.format(idx, len(all_patches)))
            img = io.imread(patch)
            if thres_saturation(img, t=15):
                if img.shape[0] != patch_size or img.shape[1] != patch_size:
                    img = transform.resize(img, (patch_size, patch_size))
                    os.remove(patch)
                    io.imsave(patch, img_as_ubyte(img))
            else:
                os.remove(patch)
        
                
                
if __name__ == '__main__':
    warnings.simplefilter('ignore')
    parser = argparse.ArgumentParser(description='Crop the WSIs into patches')
    parser.add_argument('--num_worker', type=int, default=4, help='Number of threads for parallel processing, too large may result in errors')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap pixels between adjacent patches')
    parser.add_argument('--patch_size', type=int, default=224, help='Patch size')
    args = parser.parse_args()

    print('Cropping patches, this could take a while for big datasets, please be patient')
    step = args.patch_size - args.overlap
    path_base = ('WSI/tcga-lung')
    out_base = ('WSI/tcga-lung/pyramid')
    all_slides = glob.glob(join(path_base, '*/*/*.svs')) 
    slide_to_patch(out_base, all_slides, args.patch_size, args.overlap, args.num_worker)