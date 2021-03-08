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
                        
def slide_to_patch(out_base, img_slides, patch_size=224, overlap=0, num_worker=4):
    makedirs(out_base, exist_ok=True)
    for s in tqdm(range(len(img_slides))):
        img_slide = img_slides[s]
        img_name = img_slide.split(path.sep)[-1].split('.')[0]
        bag_path = join(out_base, img_name)
        makedirs(bag_path, exist_ok=True)
        img = slide.OpenSlide(img_slide)
        dimension = img.level_dimensions[1] # given as width, height
        thumbnail = np.array(img.get_thumbnail((int(dimension[0])/7, int(dimension[1])/7)))[..., :3]
        io.imsave(join('../../../test/thumbnails', img_name + ".png"), img_as_ubyte(thumbnail))
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
    parser = argparse.ArgumentParser(description='Generate patches from testing slides')
    path_base = ('test/input')
    out_base = ('test/patches')
    makedirs('test/thumbnails', exist_ok=True)
    all_slides = glob.glob(join(path_base, '*.svs'))
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--num_worker', type=int, default=4)
    args = parser.parse_args()
    
    print('Cropping patches, please be patient')
    slide_to_patch(out_base, all_slides, args.patch_size, args.overlap, args.num_worker)