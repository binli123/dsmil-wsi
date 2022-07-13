import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import warnings

class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        path = self.files_list[idx]
        img = Image.open(path)
        img_name = path.split(os.sep)[-1]
        img_pos = np.asarray([int(img_name.split('.')[0].split('_')[0]), int(img_name.split('.')[0].split('_')[1])]) # row, col
        sample = {'input': img, 'position': img_pos}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        sample['input'] = img
        return sample
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def test(args, bags_list, milnet):
    milnet.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        feats_list = []
        pos_list = []
        classes_list = []
        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg'))
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                patch_pos = batch['position']
                feats, classes = milnet.i_classifier(patches)
                feats = feats.cpu().numpy()
                classes = classes.cpu().numpy()
                feats_list.extend(feats)
                pos_list.extend(patch_pos)
                classes_list.extend(classes)
            pos_arr = np.vstack(pos_list)
            feats_arr = np.vstack(feats_list)
            classes_arr = np.vstack(classes_list)
            bag_feats = torch.from_numpy(feats_arr).cuda()
            ins_classes = torch.from_numpy(classes_arr).cuda()
            bag_prediction, A, _ = milnet.b_classifier(bag_feats, ins_classes)
            bag_prediction = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
            color = [0, 0, 0]
            if bag_prediction >= args.thres_tumor:
                print(bags_list[i] + ' is detected as malignant')
                color = [1, 0, 0]
                attentions = A
            else:
                attentions = A
                print(bags_list[i] + ' is detected as benign')
            color_map = np.zeros((np.amax(pos_arr, 0)[0]+1, np.amax(pos_arr, 0)[1]+1, 3))
            attentions = attentions.cpu().numpy()
            attentions = exposure.rescale_intensity(attentions, out_range=(0, 1))
            for k, pos in enumerate(pos_arr):
                tile_color = np.asarray(color) * attentions[k]
                color_map[pos[0], pos[1]] = tile_color
            slide_name = bags_list[i].split(os.sep)[-1]
            color_map = transform.resize(color_map, (color_map.shape[0]*32, color_map.shape[1]*32), order=0)
            io.imsave(os.path.join('test-c16', 'output', slide_name+'.png'), img_as_ubyte(color_map))        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=1, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--thres_tumor', type=float, default=0.1964)
    args = parser.parse_args()
    
    resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    state_dict_weights = torch.load(os.path.join('test-c16', 'weights', 'embedder.pth'))
    new_state_dict = OrderedDict()
    for i in range(4):
        state_dict_weights.popitem()
    state_dict_init = i_classifier.state_dict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v
    i_classifier.load_state_dict(new_state_dict, strict=False)
    state_dict_weights = torch.load(os.path.join('test-c16', 'weights', 'aggregator.pth'))
    state_dict_weights["i_classifier.fc.weight"] = state_dict_weights["i_classifier.fc.0.weight"]
    state_dict_weights["i_classifier.fc.bias"] = state_dict_weights["i_classifier.fc.0.bias"]
    milnet.load_state_dict(state_dict_weights, strict=False)
    
    bags_list = glob.glob(os.path.join('test-c16', 'patches', '*'))
    os.makedirs(os.path.join('test-c16', 'output'), exist_ok=True)
    test(args, bags_list, milnet)