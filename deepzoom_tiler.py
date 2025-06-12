import json
from multiprocessing import Process, JoinableQueue
import argparse
import os
import re
import shutil
import sys
import glob
import numpy as np
import math
from unicodedata import normalize
from skimage import io
from skimage.color import rgb2hsv
from skimage.util import img_as_ubyte
from skimage import filters
from PIL import Image, ImageFilter, ImageStat
import pandas as pd
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None

import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator

os.chdir("/ssd_scratch/karan.p/")
VIEWER_SLIDE_NAME = 'slide'

class TileWorker(Process):
    """A child process that generates and writes tiles."""

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds,
                quality, threshold):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._threshold = threshold
        self._slide = None

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            try:
                tile = dz.get_tile(level, address)
                edge = tile.filter(ImageFilter.FIND_EDGES)
                edge = ImageStat.Stat(edge).sum
                edge = np.mean(edge)/(self._tile_size**2)
                w, h = tile.size

                # gray_img = ImageOps.grayscale(dz.get_tile(level, address))
                # gray_img_np = np.array(gray_img)
                # laplacian_var = cv2.Laplacian(gray_img_np, cv2.CV_64F).var()

                if edge > self._threshold: #  and laplacian_var > 10:
                    if not (w==self._tile_size and h==self._tile_size):
                        tile = tile.resize((self._tile_size, self._tile_size))
                    tile.save(outfile, quality=self._quality)
            except:
                print("Error occured!!")
                pass
            self._queue.task_done()
            

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)


class DeepZoomImageTiler(object):
    """Handles generation of tiles and metadata for a single image."""

    def __init__(self, dz, basename, target_levels, mag_base, format, associated, queue):
        self._dz = dz
        self._basename = basename
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._target_levels = target_levels
        self._mag_base = int(mag_base)

    def run(self):
        self._write_tiles()

    def _write_tiles(self):
        target_levels = [self._dz.level_count-i-1 for i in self._target_levels]
        mag_list = [int(self._mag_base/2**i) for i in self._target_levels]
        mag_idx = 0
        for level in range(self._dz.level_count):
            if not (level in target_levels):
                continue
            temp_path = self._basename +  "_files" + args.temp 
            tiledir = os.path.join(temp_path , str(mag_list[mag_idx]))
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)
            cols, rows = self._dz.level_tiles[level]
            for row in range(rows):
                for col in range(cols):
                    tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                    col, row, self._format))
                    if not os.path.exists(tilename):
                        self._queue.put((self._associated, level, (col, row),
                                    tilename))
                    self._tile_done()
            mag_idx += 1

    def _tile_done(self):
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 10 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                    self._associated or 'slide', count, total),
                    end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, basename, mag_levels, base_mag, objective, format, tile_size, overlap,
                limit_bounds, quality, workers, threshold):
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._mag_levels = mag_levels
        self._base_mag = base_mag
        self._objective = objective
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._dzi_data = {}
        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                        limit_bounds, quality, threshold).start()

    def run(self):
        self._run_image()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                    limit_bounds=self._limit_bounds)
        
        MAG_BASE = self._slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if MAG_BASE is None:
            MAG_BASE = self._objective
        first_level = int(math.log2(float(MAG_BASE)/self._base_mag)) # raw / input, 40/20=2, 40/40=0
        target_levels = [i+first_level for i in self._mag_levels] # levels start from 0
        target_levels.reverse()
        
        tiler = DeepZoomImageTiler(dz, basename, target_levels, MAG_BASE, self._format, associated,
                    self._queue)
        tiler.run()

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()

def nested_patches(img_slide, out_base, level=(0,), ext='jpeg'):
    print('\n Organizing patches')
    img_name = img_slide.split(os.sep)[-1].split('.')[0]
    img_class = img_slide.split(os.sep)[2]
    n_levels = len(glob.glob('WSI_temp_files' + args.temp+'/*'))
    bag_path = os.path.join(out_base, img_class, img_name)
    os.makedirs(bag_path, exist_ok=True)
    if len(level)==1:
        patches = glob.glob(os.path.join('WSI_temp_files' + args.temp, '*', '*.'+ext))
        for i, patch in enumerate(patches):
            patch_name = patch.split(os.sep)[-1]
            shutil.move(patch, os.path.join(bag_path, patch_name))
            sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(patches)))
        print('Done.')
    else:
        level_factor = 2**int(level[1]-level[0])
        levels = [int(os.path.basename(i)) for i in glob.glob(os.path.join('WSI_temp_files' + args.temp, '*'))]
        levels.sort()
        low_patches = glob.glob(os.path.join('WSI_temp_files' + args.temp, str(levels[0]), '*.'+ext))
        for i, low_patch in enumerate(low_patches):
            low_patch_name = low_patch.split(os.sep)[-1]
            shutil.move(low_patch, os.path.join(bag_path, low_patch_name))
            low_patch_folder = low_patch_name.split('.')[0]
            high_patch_path = os.path.join(bag_path, low_patch_folder)
            os.makedirs(high_patch_path, exist_ok=True)
            low_x = int(low_patch_folder.split('_')[0])
            low_y = int(low_patch_folder.split('_')[1])
            high_x_list = list( range(low_x*level_factor, (low_x+1)*level_factor) )
            high_y_list = list( range(low_y*level_factor, (low_y+1)*level_factor) )
            for x_pos in high_x_list:
                for y_pos in high_y_list:
                    high_patch = glob.glob(os.path.join('WSI_temp_files' + args.temp, str(levels[1]), '{}_{}.'.format(x_pos, y_pos)+ext))
                    if len(high_patch)!=0:
                        high_patch = high_patch[0]
                        shutil.move(high_patch, os.path.join(bag_path, low_patch_folder, high_patch.split(os.sep)[-1]))
            try:
                os.rmdir(os.path.join(bag_path, low_patch_folder))
                os.remove(low_patch)
            except:
                pass
            sys.stdout.write('\r Patch [%d/%d]' % (i+1, len(low_patches)))
        print('Done.')

if __name__ == '__main__':
    Image.MAX_IMAGE_PIXELS = None
    parser = argparse.ArgumentParser(description='Patch extraction for WSI')
    parser.add_argument('-d', '--dataset', type=str, default='lung_tcga_tumor_patch', help='Dataset name')
    parser.add_argument('-e', '--overlap', type=int, default=0, help='Overlap of adjacent tiles [0]')
    parser.add_argument('-f', '--format', type=str, default='jpeg', help='Image format for tiles [jpeg]')
    parser.add_argument('-path', '--path', type=str, default='/ssd_scratch/karan.p/lung_tcga_tumor', help='WSI path')
    parser.add_argument('-v', '--slide_format', type=str, default='svs', help='Image format for tiles [svs]')
    parser.add_argument('-j', '--workers', type=int, default=16, help='Number of worker processes to start [4]')
    parser.add_argument('-q', '--quality', type=int, default=70, help='JPEG compression quality [70]')
    parser.add_argument('-s', '--tile_size', type=int, default=224, help='Tile size [224]')
    parser.add_argument('-b', '--base_mag', type=float, default=20, help='Maximum magnification for patch extraction [20]')
    parser.add_argument('-m', '--magnifications', type=int, nargs='+', default=(2,3), help='Levels for patch extraction [0]')
    parser.add_argument('-o', '--objective', type=float, default=20, help='The default objective power if metadata does not present [20]')
    parser.add_argument('-t', '--background_t', type=int, default=15, help='Threshold for filtering background [15]')
    parser.add_argument('-temp', '--temp', type=int, default=0, help='temp folder sufix')
    parser.add_argument('-st', '--start', type=int, default=-1, help='temp folder sufix')
    parser.add_argument('-ed', '--end', type=int, default=-1, help='temp folder sufix')
    parser.add_argument('-lable_csv', '--lable_csv', type= str, default="/home/karan.padariya/CLAM/dataset_csv/updated_tcga-LUAD&LUSC_updated_modified.csv")
    
    # python deepzoom_tiler.py --dataset "ORCHID" --temp 0 --start 0 --end 10  --slide_format ".png" --magnifications (0,)

    args = parser.parse_args()
    args.temp = str(args.temp)

    levels = tuple(sorted(args.magnifications))
    assert len(levels)<=2, 'Only 1 or 2 magnifications are supported!'
    path_base = args.path #'/ssd_scratch/karan.p/HDD1' #os.path.join('WSI', args.dataset)
    if len(levels) == 2:
        out_base = os.path.join('/ssd_scratch/karan.p/', args.dataset, 'pyramid')
    else:
        out_base = os.path.join('/ssd_scratch/karan.p/', args.dataset, 'single')
    print("Path: ", os.path.join(path_base, '/*/*/*.' + args.slide_format))
    all_slides = glob.glob(os.path.join(path_base, '/*.'+args.slide_format)) +  glob.glob(os.path.join(path_base, '*/*/*.'+args.slide_format))
    all_slides = glob.glob(path_base+'/*.'+args.slide_format) 

    df = pd.read_csv(args.lable_csv)
    slides = df['slide_id'].to_list()
    qual = df['quality'].to_list()

    slides_ = [slide for slide in slides ]
    slides = slides_
    all_slides = []

    for idx, slide in enumerate(slides):
        slide_path = os.path.join(path_base, os.path.basename(slide) + "." + args.slide_format)
        
        if os.path.isfile(slide_path) and "1" in str(qual[idx]):
            all_slides.append(slide_path)

    if args.start<0:
       args.start = 0
    if args.end<0:
        args.end = len(all_slides)

    # pos-i_pos-j -> x, y
    for idx in range(args.start, args.end):
        c_slide = all_slides[idx]
        filename = os.path.basename(c_slide)
        name, extension = os.path.splitext(filename)
        if len(levels) == 2:
            full_path = "/ssd_scratch/karan.p/"+args.dataset+"/pyramid/karan.p/"+name.split(".")[0]
        else:
            full_path = "/ssd_scratch/karan.p/"+args.dataset+"/single/karan.p/"+name

        if os.path.exists(full_path):
            print(f'{name} already exists.')
            continue
        else:
            print(f'Processing: {filename}')
        print('Process slide {}/{}'.format(idx+1, len(all_slides)))
        DeepZoomStaticTiler(c_slide, 'WSI_temp', levels, args.base_mag, args.objective, args.format, args.tile_size, args.overlap, True, args.quality, args.workers, args.background_t).run()
        nested_patches(c_slide, out_base, levels, ext=args.format)
        shutil.rmtree('WSI_temp_files' + args.temp) 
    print('Patch extraction done for {} slides.'.format(args.end - args.start))