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

def crop_slide(img, save_slide_path, position=(0, 0), step=(0, 0), patch_size=224, multiscale=0): # position given as (x, y) at 5x scale
    if multiscale == 1:
        if int(np.floor(float(img.properties['openslide.mpp-x'])*10)) == 2:
            img_40x = img.read_region((position[0]*2 * 4, position[1]*2 * 4), 0, (patch_size*2 * 4, patch_size*2 * 4))
            img_40x = np.array(img_40x)[..., :3]
            img_5x = transform.resize(img_40x, (img_40x.shape[0] // 8, img_40x.shape[0] // 8), order=1,  anti_aliasing=False)
            if thres_saturation(img_5x, 25):
                patch_name = "{}_{}".format(step[0], step[1])
                makedirs(join(save_slide_path, patch_name), exist_ok=True)
                io.imsave(join(save_slide_path, patch_name + ".jpg"), img_as_ubyte(img_5x))       
                img_20x = transform.resize(img_40x, (img_40x.shape[0] // 2, img_40x.shape[0] // 2), order=1,  anti_aliasing=False)
                img_20x_subs = np.squeeze(view_as_windows(img_20x, (patch_size, patch_size, 3), patch_size))
                for j in range(img_20x_subs.shape[0]):
                    for i in range(img_20x_subs.shape[1]):
                        if thres_saturation(img_20x_subs[j, i], 25):
                            io.imsave(join(save_slide_path, patch_name, "{}_{}.jpg".format(j, i)), img_as_ubyte(img_20x_subs[j, i])) # index given as (rows, column)
        else:
            img_20x = img.read_region((position[0] * 4, position[1] * 4), 0, (patch_size * 4, patch_size * 4))
            img_20x = np.array(img_20x)[..., :3]
            img_5x = transform.resize(img_20x, (img_20x.shape[0] // 4, img_20x.shape[0] // 4), order=1,  anti_aliasing=False)
            if thres_saturation(img_5x, 25):
                patch_name = "{}_{}".format(step[0], step[1])
                makedirs(join(save_slide_path, patch_name), exist_ok=True)
                io.imsave(join(save_slide_path, patch_name + ".jpg"), img_as_ubyte(img_5x))       
                img_20x_subs = np.squeeze(view_as_windows(img_20x, (patch_size, patch_size, 3), patch_size))
                for j in range(img_20x_subs.shape[0]):
                    for i in range(img_20x_subs.shape[1]):
                        if thres_saturation(img_20x_subs[j, i], 25):
                            io.imsave(join(save_slide_path, patch_name, "{}_{}.jpg".format(j, i)), img_as_ubyte(img_20x_subs[j, i])) # index given as (rows, column)
    else:
        if int(np.floor(float(img.properties['openslide.mpp-x'])*10)) == 2:
            img = img.read_region((position[0] * 4, position[1] * 4), 1, (patch_size, patch_size))
            img = np.array(img)[..., :3]
            if thres_saturation(img, 30):
                patch_name = "{}_{}".format(step[0], step[1])
                io.imsave(join(save_slide_path, patch_name + ".jpg"), img_as_ubyte(img))
        else:
            img_20x = img.read_region((position[0] * 2, position[1] * 2), 0, (patch_size * 2, patch_size * 2))
            img_20x = np.array(img_20x)[..., :3]
            img = transform.resize(img_20x, (img_20x.shape[0] // 2, img_20x.shape[0] // 2), order=1,  anti_aliasing=False)
            if thres_saturation(img, 30):
                patch_name = "{}_{}".format(step[0], step[1])
                io.imsave(join(save_slide_path, patch_name + ".jpg"), img_as_ubyte(img))
                        
def slide_to_patch(out_base, img_slides, patch_size, step_size, multiscale):
    makedirs(out_base, exist_ok=True)
    for s in tqdm(range(len(img_slides))):
        img_slide = img_slides[s]
        img_name = img_slide.split(path.sep)[-1].split('.')[0]
        img_class = img_slide.split(path.sep)[2]
        bag_path = join(out_base, img_class, img_name)
        makedirs(bag_path, exist_ok=True)
        img = slide.OpenSlide(img_slide)
        if int(np.floor(float(img.properties['openslide.mpp-x'])*10)) == 2:
            dimension = img.level_dimensions[0] # given as width, height
            dimension_5x = (int(dimension[0] / 8), int(dimension[1] / 8))
        else:
            dimension = img.level_dimensions[0]
            dimension_5x = (int(dimension[0] / 4), int(dimension[1] / 4))
        if multiscale == 1:
            step_y_max = int(np.floor(dimension_5x[1]/step_size)) # rows
            step_x_max = int(np.floor(dimension_5x[0]/step_size)) # columns
        else:
            if int(np.floor(float(img.properties['openslide.mpp-x'])*10)) == 2:
                dimension = img.level_dimensions[1]
                step_y_max = int(np.floor(dimension[1]/step_size)) # rows
                step_x_max = int(np.floor(dimension[0]/step_size)) # columns
            else:
                dimension = img.level_dimensions[0]
                step_y_max = int(np.floor(dimension[1]/(step_size*2))) # rows
                step_x_max = int(np.floor(dimension[0]/(step_size*2))) # columns
        for j in range(step_y_max):
            for i in range(step_x_max):
                crop_slide(img, bag_path, (i*step_size, j*step_size), step=(j, i), patch_size=patch_size, multiscale=multiscale)
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop the WSIs into patches')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads for parallel processing, too large may result in errors')
    parser.add_argument('--overlap', type=int, default=0, help='Overlap pixels between adjacent patches')
    parser.add_argument('--patch_size', type=int, default=224, help='Patch size')
    parser.add_argument('--multiscale', type=int, default=0, help='Multiscale dataset 20x/5x or 10x (1 or 0).')
    parser.add_argument('--dataset', type=str, default='TCGA-lung', help='Dataset folder name')
    args = parser.parse_args()

    print('Cropping patches, this could take a while for big dataset, please be patient')
    step = args.patch_size - args.overlap
    path_base = join('WSI', args.dataset)
    if args.multiscale == 1:
        out_base = join('WSI', args.dataset, 'pyramid')
    else:
        out_base = join('WSI', args.dataset, 'single')
    all_slides = glob.glob(join(path_base, '*/*.svs')) +  glob.glob(join(path_base, '*/*/*.svs'))
    each_thread = int(np.floor(len(all_slides)/args.num_threads))
    threads = []
    for i in range(args.num_threads):
        if i < (args.num_threads-1):
            t = threading.Thread(target=slide_to_patch, args=(out_base, all_slides[each_thread*i:each_thread*(i+1)], args.patch_size, step, args.multiscale))
        else:
            t = threading.Thread(target=slide_to_patch, args=(out_base, all_slides[each_thread*i:], args.patch_size, step, args.multiscale))
        threads.append(t)

    for thread in threads:
        thread.start()