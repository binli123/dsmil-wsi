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



class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        sample = {'input': img}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        return {'input': img} 
    
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
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def compute_feats(args, bags_list, i_classifier, save_path=None):
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        feats_list = []
        if  or args.magnification == '20x':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*/*.jpg'))
        if args.magnification == '5x' or args.magnification == '10x':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg'))
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda() 
                feats, classes = i_classifier(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
        df = pd.DataFrame(feats_list)
        os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
        df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
        sys.stdout.write('\r Computed: {}/{}'.format(i+1, num_bags))
        

def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')
    parser.add_argument('--num_feats', default=512, type=int, help='Feature size')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--dataset', default='wsi-tcga-lung', type=str, help='Nanme of dataset')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone')
    parser.add_argument('--magnification', default='20x', type=str, help='Magnification to compute features')
    parser.add_argument('--weights', default=None, type=str, help='Folder of the pretrained weights, simclr/runs/*')
    args = parser.parse_args()
    
    if args.backbone == 'resnet18':
        resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 512
    if args.backbone == 'resnet34':
        resnet = models.resnet34(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 512
    if args.backbone == 'resnet50':
        resnet = models.resnet50(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 2048
    if args.backbone == 'resnet101':
        resnet = models.resnet101(pretrained=False, norm_layer=nn.InstanceNorm2d)
        num_feats = 2048
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    i_classifier = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()
    
    if args.weights is not None:
        weight_path = os.path.join('simclr', 'runs', args.weights, 'checkpoints', 'model.pth')
    else:
        weight_path = glob.glob('simclr/runs/*/checkpoints/*.pth')[-1]
    state_dict_weights = torch.load(weight_path)

    try:
        state_dict_weights.pop('module.l1.weight')
        state_dict_weights.pop('module.l1.bias')
        state_dict_weights.pop('module.l2.weight')
        state_dict_weights.pop('module.l2.bias')
    except:
        state_dict_weights.pop('l1.weight')
        state_dict_weights.pop('l1.bias')
        state_dict_weights.pop('l2.weight')
        state_dict_weights.pop('l2.bias')
    state_dict_init = i_classifier.state_dict()
    new_state_dict = OrderedDict()
    for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
        name = k_0
        new_state_dict[name] = v
    i_classifier.load_state_dict(new_state_dict, strict=False)
    
    if args.dataset == 'wsi-tcga-lung':
        bags_path = os.path.join('WSI', 'TCGA-lung', 'pyramid', '*', '*')
    if args.dataset == 'wsi-tcga-lung-single':
        bags_path = os.path.join('WSI', 'TCGA-lung', 'single', '*', '*')
    feats_path = os.path.join('datasets', args.dataset)
    os.makedirs(feats_path, exist_ok=True)
    bags_list = glob.glob(bags_path)
    compute_feats(args, bags_list, i_classifier, feats_path)
    
if __name__ == '__main__':
    main()