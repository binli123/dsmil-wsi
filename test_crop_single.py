### TCGA
import openslide as slide
from PIL import Image
import numpy as np
from skimage import data, io, transform
from skimage.color import rgb2gray, rgb2hsv
from skimage.util import img_as_ubyte, view_as_windows
from skimage import img_as_ubyte
from os import listdir, mkdir, path, makedirs
from os.path import join 
import time, sys, warnings, glob
import threading
from tqdm import tqdm
import argparse
warnings.simplefilter('ignore')

def thres_saturation(img, t=15):
    # typical t = 15
    img = rgb2hsv(img)
    h, w, c = img.shape
    sat_img = img[:, :, 1]
    sat_img = img_as_ubyte(sat_img)
    ave_sat = np.sum(sat_img) / (h * w)
    return ave_sat >= t

def crop_slide(img, save_slide_path, position=(0, 0), step=(0, 0), patch_size=224): # position given as (x, y) 
        img = img.read_region((position[0] * 4, position[1] * 4), 1, (patch_size, patch_size))
        img = np.array(img)[..., :3]
        if thres_saturation(img, 30):
            patch_name = "{}_{}".format(step[0], step[1])
            io.imsave(join(save_slide_path, patch_name + ".jpg"), img_as_ubyte(img))       
                        
def slide_to_patch(out_base, img_slides, step, folder='test'):
    makedirs(out_base, exist_ok=True)
    patch_size = 224
    step_size = step
    for s in range(len(img_slides)):
        img_slide = img_slides[s]
        img_name = img_slide.split(path.sep)[-1].split('.')[0]
        bag_path = join(out_base, img_name)
        makedirs(bag_path, exist_ok=True)
        img = slide.OpenSlide(img_slide)
        dimension = img.level_dimensions[1] # given as width, height
        if folder=='test':
            thumbnail = np.array(img.get_thumbnail((int(dimension[0])/7, int(dimension[1])/7)))[..., :3]
        else:
            thumbnail = np.array(img.get_thumbnail((int(dimension[0])/28, int(dimension[1])/28)))[..., :3]
        io.imsave(join(folder, 'thumbnails', img_name + ".png"), img_as_ubyte(thumbnail))        
        step_y_max = int(np.floor(dimension[1]/step_size)) # rows
        step_x_max = int(np.floor(dimension[0]/step_size)) # columns
        for j in range(step_y_max): # rows
            for i in range(step_x_max): # columns
                crop_slide(img, bag_path, (i*step_size, j*step_size), step=(j, i), patch_size=patch_size)
            sys.stdout.write('\r Cropped: {}/{} -- {}/{}'.format(s+1, len(img_slides), j+1, step_y_max))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate patches from testing slides')
    parser.add_argument('--dataset', type=str, default='tcga', help='Dataset name [tcga]')
    args = parser.parse_args()
    if args.dataset == 'tcga':
        path_base = ('test/input')
        out_base = ('test/patches')
        folder = 'test'
        makedirs('test/thumbnails', exist_ok=True)
    elif args.dataset == 'c16':
        path_base = ('test-c16/input')
        out_base = ('test-c16/patches')
        folder = ('test-c16')
        makedirs('test-c16/thumbnails', exist_ok=True)
    all_slides = glob.glob(join(path_base, '*.svs')) + glob.glob(join(path_base, '*.tif'))
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--patch_size', type=int, default=224)
    args = parser.parse_args()

    print('Cropping patches, please be patient')
    step = args.patch_size - args.overlap
    slide_to_patch(out_base, all_slides, step, folder)