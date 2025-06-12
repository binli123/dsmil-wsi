import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle
from functools import partial
import cv2
# from main_downstream_linear_dinov2 import QuickGELU
import models_dinov2
import timm

os.chdir("/ssd_scratch/karan.p")


class QuickGELU(torch.nn.Module):
   def forward(self, x: torch.Tensor):
       return x * torch.sigmoid(1.702 * x)


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
        img = img.resize((224,224))
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

class Pix2Pix(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load('/home/karan.padariya/CLAM/Pix2Pix_stain_norm_model/model_11_2400.pth', map_location=self.device,weights_only=False)
        self.model = self.model.to(self.device).float()
        self.model.eval()

    def __call__(self, I):
        with torch.no_grad():
            original_size = I['input'].size[:2]  # (height, width)
            I_resized = cv2.resize(np.array(I['input']), (256,256))
            I_resized = I_resized / 127.5 - 1.0 
            I_resized = np.expand_dims(I_resized, 0)

            if len(I_resized.shape) == 4: 
                I_tensor = torch.stack([torch.tensor(img, dtype=torch.float32).to(self.device) for img in I_resized])
            else:
				# If it's a single image, apply to_tensor directly
                I_tensor = torch.tensor(I_resized, dtype=torch.float32).to(self.device)
            output = self.model(I_tensor)
			
            prediction_resized = output.cpu().numpy()
            prediction_resized = cv2.resize(prediction_resized[0], original_size)      
            return {'input': prediction_resized}

def get_eval_transforms_stain_net():
	trsforms= []
	stain_norm_transform  =  Pix2Pix()
	# if target_img_size > 0:
	# 	trsforms.append(transforms.Resize(target_img_size))
	
	trsforms.append(stain_norm_transform)
	trsforms.append(ToTensor())
	# trsforms.append(transforms.Normalize(mean, std))
	trsforms = Compose(trsforms)
	return trsforms

def bag_dataset(args, csv_file_path):

    # stain_norm = False
    if args.stain_norm:
        transform = get_eval_transforms_stain_net()
    else:
        transform=Compose([ToTensor()])  

    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=transform)
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def compute_feats(args, bags_list, i_classifier, save_path=None, magnification='single'):
    i_classifier.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    for i in range(0, num_bags):
        feats_list = []
        if magnification=='single' or magnification=='low':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        elif magnification=='high':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.jpg')) + glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.jpeg'))
            print()
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda() 
                feats, classes = i_classifier(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, iteration+1, len(dataloader)))
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            df = pd.DataFrame(feats_list)
            os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
            df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
        
def compute_tree_feats(args, bags_list, embedder_low, embedder_high, save_path=None):
    embedder_low.eval()
    embedder_high.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    with torch.no_grad():
        for i in range(0, num_bags): 
            filename = os.path.basename(bags_list[i])

            path = '/ssd_scratch/karan.p/datasets/'+args.dataset+'/karan.p'

            # Check if the file exists in the specified path
            file_path = os.path.join(path, filename)
            if os.path.exists(file_path+'.csv'):
                print(f"File '{filename}.csv' already exists at {path}.")
                continue
            else:
                print(f"Processing {filename}")

            low_patches = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg')) + glob.glob(os.path.join(bags_list[i], '*.tiff'))
            feats_list = []
            feats_list = []
            feats_tree_list = []
            dataloader, bag_size = bag_dataset(args, low_patches)
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                feats, classes = embedder_low(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
            for idx, low_patch in enumerate(low_patches):
                high_folder = os.path.dirname(low_patch) + os.sep + os.path.splitext(os.path.basename(low_patch))[0]
                high_patches = glob.glob(high_folder+os.sep+'*.jpg') + glob.glob(high_folder+os.sep+'*.jpeg') + glob.glob(high_folder+os.sep+'*.tiff')
                if len(high_patches) == 0:
                    pass
                else:
                    for high_patch in high_patches:
                        img = Image.open(high_patch)
                        img = VF.to_tensor(img).float().cuda()
                        feats, classes = embedder_high(img[None, :])
                        
                        if args.tree_fusion == 'fusion':
                            feats = feats.cpu().numpy()+0.25*feats_list[idx]
                        elif args.tree_fusion == 'cat':
                            feats = np.concatenate((feats.cpu().numpy(), feats_list[idx][None, :]), axis=-1)
                        else:
                            raise NotImplementedError(f"{args.tree_fusion} is not an excepted option for --tree_fusion. This argument accepts 2 options: 'fusion' and 'cat'.")
                        
                        feats_tree_list.extend(feats)
                sys.stdout.write('\r Computed: {}/{} -- {}/{}'.format(i+1, num_bags, idx+1, len(low_patches)))
            if len(feats_tree_list) == 0:
                print('No valid patch extracted from: ' + bags_list[i])
            else:
                df = pd.DataFrame(feats_tree_list)
                os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
                df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
            print('\n')            

def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,1,2,3), help='GPU ID(s) [0]')
    parser.add_argument('--backbone', default='conch', type=str, help='Embedder backbone [resnet18]')
    parser.add_argument('--norm_layer', default='batch', type=str, help='Normalization layer [instance]')
    parser.add_argument('--magnification', default='tree', type=str, help='Magnification to compute features. Use `tree` for multiple magnifications. Use `high` if patches are cropped for multiple resolution and only process higher level, `low` for only processing lower level.')
    parser.add_argument('--weights', default='ImageNet', type=str, help='Folder of the pretrained weights, simclr/runs/*')
    parser.add_argument('--weights_high', default='ImageNet', type=str, help='Folder of the pretrained weights of high magnification, FOLDER < `simclr/runs/[FOLDER]`')
    parser.add_argument('--weights_low', default='ImageNet', type=str, help='Folder of the pretrained weights of low magnification, FOLDER <`simclr/runs/[FOLDER]`')
    parser.add_argument('--tree_fusion', default='cat', type=str, help='Fusion method for high and low mag features in a tree method [cat|fusion]')
    parser.add_argument('--dataset', default='lung_tcga_tumor', type=str, help='Dataset folder name [TCGA-lung-single]')
    parser.add_argument('--stain_norm', default=False, type=bool, help='apply pix2pix stain normalization or not')
    parser.add_argument('--csv_file', default='/home/karan.padariya/CLAM/dataset_csv/updated_tcga-LUAD&LUSC_updated_modified.csv', help = "path of csv file containing slide_id and its label")
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)

    if args.norm_layer == 'instance':
        norm=nn.InstanceNorm2d
        pretrain = False
    elif args.norm_layer == 'batch':  
        norm=nn.BatchNorm2d
        if args.weights == 'ImageNet':
            pretrain = True
        else:
            pretrain = False

    if args.backbone == 'resnet18':
        resnet = models.resnet18(pretrained=pretrain, norm_layer=norm)
        num_feats = 512
    if args.backbone == 'conch':
        from conch.open_clip_custom import create_model_from_pretrained
        resnet, _ = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="key")
        resnet.forward = partial(resnet.encode_image, proj_contrast=False, normalize=False)
        num_feats = 512

    if args.backbone =='uni':
        resnet = timm.create_model("vit_large_patch16_224",
                            init_values=1e-5, 
                            num_classes=0, 
                            dynamic_img_size=True)
        resnet.load_state_dict(torch.load("/home/karan.padariya/CLAM/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/uni.bin", map_location="cpu"), strict=True)
        num_feats = 1024

    if args.backbone == 'resnet34':
        resnet = models.resnet34(pretrained=pretrain, norm_layer=norm)
        num_feats = 512

    # if args.backbone == 'resnet50':
    #     resnet = models.resnet50(pretrained=pretrain, norm_layer=norm)
    #     num_feats = 512

    if args.backbone == 'resnet50_dino':
        weight_path = "/ssd_scratch/karan.p/log/DINOv2_training_our/checkpoint.pth"
        resnet = models.resnet50(pretrained=pretrain, norm_layer=norm)
        checkpoint = torch.load(weight_path)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove prefix like 'module.' if model was trained with DataParallel or DistributedDataParallel
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load the state dictionary into the model
        resnet.load_state_dict(state_dict, strict=False)
        num_feats = 2048
    if args.backbone == 'resnet101':
        resnet = models.resnet101(pretrained=pretrain, norm_layer=norm)
        num_feats = 2048

    if args.backbone == 'vit_base':
        import_student = getattr(models_dinov2, args.backbone)
        model = import_student(img_size=224,
            patch_size=14,
            init_values=1.0,
            ffn_layer='mlp',
            block_chunks=0,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1)

        checkpoint = torch.load("/ssd_scratch/karan.p/log/DINOv2_training_our/checkpoint.pth", map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'teacher' in checkpoint:
            state_dict = checkpoint['teacher']
        else:
            state_dict = checkpoint

        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            if 'student' in k:
                new_state_dict[k.replace("student.backbone.", "")] = v
        args.quickgelu = False
        if args.quickgelu:
            print('replacing gelu in {} layers to quickgelu'.format(len(model.blocks)))
            for i in range(len(model.blocks)):
                model.blocks[i].mlp.act = QuickGELU()

        msg = model.load_state_dict(new_state_dict, strict=False)
        print(msg)
        resnet = model
        num_feats = 768

    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    
    if args.magnification == 'tree' and args.weights_high != None and args.weights_low != None:
        i_classifier_h = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()
        i_classifier_l = mil.IClassifier(copy.deepcopy(resnet), num_feats, output_class=args.num_classes).cuda()
        
        if args.weights_high == 'ImageNet' or args.weights_low == 'ImageNet' or args.weights== 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                raise ValueError('Please use batch normalization for ImageNet feature')
        elif args.backbone == 'resnet18':
            pass
        else:
            # weight_path = os.path.join('simclr', 'runs', args.weights_high, 'checkpoints', 'model.pth')
            weight_path = "/home/karan.padariya/barlowtwins/ssd_scratch/checkpoint_bt/checkpoint.pth"
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier_h.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_h.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder-high.pth'))

            # weight_path = os.path.join('simclr', 'runs', args.weights_low, 'checkpoints', 'model.pth')
            weight_path = "/home/karan.padariya/barlowtwins/ssd_scratch/checkpoint_bt/checkpoint.pth"
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier_l.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_l.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder-low.pth'))
            print('Use pretrained features.')


    elif args.magnification == 'single' or args.magnification == 'high' or args.magnification == 'low':  
        i_classifier = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()

        if args.weights == 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                print('Please use batch normalization for ImageNet feature')
        else:
            if args.weights is not None:
                weight_path = os.path.join('simclr', 'runs', args.weights, 'checkpoints', 'model.pth')
            else:
                weight_path = glob.glob('simclr/runs/*/checkpoints/*.pth')[-1]
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.dataset), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.dataset, 'embedder.pth'))
            print('Use pretrained features.')
    
    if args.magnification == 'tree' or args.magnification == 'low' or args.magnification == 'high' :
        bags_path = os.path.join( args.dataset, 'pyramid', '*', '*')
    else:
        bags_path = os.path.join( args.dataset, 'single', '*', '*')
    feats_path = os.path.join('datasets', args.dataset)
        
    os.makedirs(feats_path, exist_ok=True)
    bags_list = glob.glob(bags_path)
    
    if args.magnification == 'tree':
        compute_tree_feats(args, bags_list, i_classifier_l, i_classifier_h, feats_path)
    else:
        compute_feats(args, bags_list, i_classifier, feats_path, args.magnification)
    n_classes = glob.glob(os.path.join('datasets', args.dataset, '*'+os.path.sep))
    n_classes = sorted(n_classes)
    all_df = []
    for i, item in enumerate(n_classes):
        bag_csvs = glob.glob(os.path.join(item, '*.csv'))
        
        label_file = pd.read_csv(args.csv_file)
        extracted_slides = [os.path.splitext(os.path.basename(bag_csv))[0]+ ".tiff" for bag_csv in bag_csvs]
        filtered_df = label_file[label_file['slide_id'].isin(extracted_slides)]
        
        slide_path = [os.path.join('datasets', args.dataset,'karan.p', slide_id[:-5]+'.csv') for slide_id in filtered_df['slide_id']]
        bag_df = pd.DataFrame(slide_path)
        # bag_df['0'] = slide_path

        labels = []
        for class_name in filtered_df['label']:
            if class_name == "ADENO":
                labels.append(0)
            elif class_name == "SQUAMOUS":
                labels.append(1)
            elif class_name == "Stage_3":
                labels.append(2)

        bag_df['label'] = labels

        bag_df.to_csv(os.path.join('datasets', args.dataset, item.split(os.path.sep)[2]+'.csv'), index=False)
        all_df.append(bag_df)
    bags_path = pd.concat(all_df, axis=0, ignore_index=True)
    bags_path = shuffle(bags_path)
    bags_path.to_csv(os.path.join('datasets', args.dataset, args.dataset+'.csv'), index=False)
    
if __name__ == '__main__':
    main()