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
    colors = [np.random.choice(range(256), size=3) for i in range(args.num_classes)]
    for i in range(0, num_bags):
        feats_list = []
        pos_list = []
        classes_list = []
        csv_file_path = glob.glob(os.path.join(bags_list[i], '*.'+args.patch_ext))
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
            if len(bag_prediction.shape)==0 or len(bag_prediction.shape)==1:
                bag_prediction = np.atleast_1d(bag_prediction)
            benign = True
            num_pos_classes = 0
            for c in range(args.num_classes):          
                if bag_prediction[c] >= args.thres[c]:
                    attentions = A[:, c].cpu().numpy()
                    num_pos_classes += 1
                    if benign: # first class detected
                        print(bags_list[i] + ' is detected as: ' + args.class_name[c])
                        colored_tiles = np.matmul(attentions[:, None], colors[c][None, :])
                    else:
                        print('and ' + args.class_name[c])          
                        colored_tiles = colored_tiles + np.matmul(attentions[:, None], colors[c][None, :])
                    benign = False # set flag
            if benign:
                print(bags_list[i] + ' is detected as: benign')
                attentions = torch.sum(A, 1).cpu().numpy()
                colored_tiles = np.matmul(attentions[:, None], colors[0][None, :]) * 0
            colored_tiles = (colored_tiles / num_pos_classes)
            colored_tiles = exposure.rescale_intensity(colored_tiles, out_range=(0, 1))
            color_map = np.zeros((np.amax(pos_arr, 0)[0]+1, np.amax(pos_arr, 0)[1]+1, 3))
            for k, pos in enumerate(pos_arr):
                color_map[pos[0], pos[1]] = colored_tiles[k]
            slide_name = bags_list[i].split(os.sep)[-1]
            color_map = transform.resize(color_map, (color_map.shape[0]*32, color_map.shape[1]*32), order=0)
            io.imsave(os.path.join(args.map_path, slide_name+'.png'), img_as_ubyte(color_map))
            if args.export_scores:
                df_scores = pd.DataFrame(A.cpu().numpy())
                pos_arr_str = [str(s) for s in pos_arr]
                df_scores['pos'] = pos_arr_str
                df_scores.to_csv(os.path.join(args.score_path, slide_name+'.csv'), index=False)
                
                
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--thres', nargs='+', type=float, default=[0.7371, 0.2752])
    parser.add_argument('--class_name', nargs='+', type=str, default=None)
    parser.add_argument('--embedder_weights', type=str, default='test/weights/embedder.pth')
    parser.add_argument('--aggregator_weights', type=str, default='test/weights/aggregator.pth')
    parser.add_argument('--bag_path', type=str, default='test/patches')
    parser.add_argument('--patch_ext', type=str, default='jpg')
    parser.add_argument('--map_path', type=str, default='test/output')
    parser.add_argument('--export_scores', type=int, default=0)
    parser.add_argument('--score_path', type=str, default='test/score')
    args = parser.parse_args()
    
    if args.embedder_weights == 'ImageNet':
        print('Use ImageNet features')
        resnet = models.resnet18(pretrained=True, norm_layer=nn.BatchNorm2d)
    else:
        resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()

    if args.embedder_weights !=  'ImageNet':
        state_dict_weights = torch.load(args.embedder_weights)
        new_state_dict = OrderedDict()
        for i in range(4):
            state_dict_weights.popitem()
        state_dict_init = i_classifier.state_dict()
        for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            name = k_0
            new_state_dict[name] = v
        i_classifier.load_state_dict(new_state_dict, strict=False)

    state_dict_weights = torch.load(args.aggregator_weights) 
    state_dict_weights["i_classifier.fc.weight"] = state_dict_weights["i_classifier.fc.0.weight"]
    state_dict_weights["i_classifier.fc.bias"] = state_dict_weights["i_classifier.fc.0.bias"]
    milnet.load_state_dict(state_dict_weights, strict=False)

    bags_list = glob.glob(os.path.join(args.bag_path, '*'))
    os.makedirs(args.map_path, exist_ok=True)
    if args.export_scores:
        os.makedirs(args.score_path, exist_ok=True)
    if args.class_name == None:
        args.class_name = ['class {}'.format(c) for c in range(args.num_classes)]
    if len(args.thres) != args.num_classes:
        raise ValueError('Number of thresholds does not match classes.')
    test(args, bags_list, milnet)