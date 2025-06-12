# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from pprint import pprint
from enum import Enum
import time
import PIL
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision import datasets as t_datasets

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from cuml.linear_model import LogisticRegression as LogRegGPU

from collections import OrderedDict
from PIL import Image
import models_dinov2


NUM_VOC_CLASSES = 20
BASE_SAVE_DIR = '/ssd_scratch/cache_features/'
DATA_CACHE_DIR = '/ssd_scratch/cache_data'
RESULT_CACHE_DIR = '/ssd_scratch/cache_result'
DATA_DIR = '/ssd_scratch/karan.p/ORCHID_patches'


class MetricMode(Enum):
    ACC = "accuracy"
    MEAN_CLS_ACC = "mean-class-accuracy"
    MEAN_AP = "mean AP"


def load_features(feat_file, normalize=True):
    data = np.load(feat_file)
    data_dict = {}
    data_dict['x_train'] = data['X_train_feature']
    data_dict['y_train'] = data['y_train']
    data_dict['x_val'] = data['X_val_feature']
    data_dict['y_val'] = data['y_val']
    data_dict['x_trainval'] = data['X_trainval_feature']
    data_dict['y_trainval'] = data['y_trainval']
    data_dict['x_test'] = data['X_test_feature']
    data_dict['y_test'] = data['y_test']

    if normalize:
        train_mean = np.mean(data_dict['x_train'], axis=0, keepdims=True)
        train_std = np.std(data_dict['x_train'], axis=0, keepdims=True)

        data_dict['x_train'] = (data_dict['x_train'] - train_mean) / train_std
        data_dict['x_val'] = (data_dict['x_val'] - train_mean) / train_std

        trainval_mean = np.mean(data_dict['x_trainval'], axis=0, keepdims=True)
        trainval_std = np.std(data_dict['x_trainval'], axis=0, keepdims=True)

        data_dict['x_trainval'] = (data_dict['x_trainval'] - trainval_mean) / trainval_std
        data_dict['x_test'] = (data_dict['x_test'] - trainval_mean) / trainval_std

    print(f'Features loaded from {feat_file}')
    return data_dict


def lbfgs_gpu(data_dict,
              max_iter=1000,
              tol=1e-12,
              metric=MetricMode.ACC):

    wd_range = torch.logspace(-6, 5, 45)

    best_acc = 0
    best_c = None

    # GPU version
    clf = LogRegGPU(max_iter=max_iter, output_type="numpy", tol=tol, linesearch_max_iter=50)

    # # CPU version
    # clf = LogRegCPU(max_iter=max_iter, solver='lbfgs', multi_class='multinomial', warm_start=True)

    start = time.time()

    def fit_and_evaluate(clf, x_train, y_train, x_test, y_test):
        if metric == MetricMode.ACC:
            clf.fit(x_train, y_train)
            test_acc = 100. * clf.score(x_test, y_test)
            return test_acc
        elif metric == MetricMode.MEAN_CLS_ACC:
            clf.fit(x_train, y_train)
            pred_test = clf.predict(x_test)
            cm = confusion_matrix(y_test, pred_test)
            cm = cm.diagonal() / cm.sum(axis=1)
            test_score = 100. * cm.mean()
            return test_score
        elif metric == MetricMode.MEAN_AP:
            num_classes = NUM_VOC_CLASSES
            aps_test = []
            for cls in range(num_classes):
                clf.fit(x_train, y_train[:, cls])
                pred_test = clf.decision_function(x_test)
                ap_test = voc_eval_cls(y_test[:, cls], pred_test)
                aps_test.append(ap_test)
            mAP_test = 100. * np.mean(aps_test)
            return mAP_test
        else:
            raise NotImplemented(f'Metric {metric} is not supported')

    for wd in wd_range:
        c = 1. / wd.item()
        clf.set_params(C=c)
        test_acc = fit_and_evaluate(clf, data_dict['x_train'], data_dict['y_train'], data_dict['x_val'], data_dict['y_val'])

        if test_acc > best_acc:
            best_acc = test_acc
            best_c = c

    # do trainval
    clf.set_params(C=best_c)
    test_acc = fit_and_evaluate(clf, data_dict['x_trainval'], data_dict['y_trainval'], data_dict['x_test'], data_dict['y_test'])
    print(f'done in {time.time() - start:.3f} seconds')
    return test_acc


def voc_ap(rec, prec):
    ap = 0.
    for t in np.linspace(0, 1, 11):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap += p / 11.
    return ap


def voc_eval_cls(y_true, y_pred):
    # get precision and recall
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    # compute average precision
    ap = voc_ap(rec, prec)
    return ap

class ORAL(Dataset):
    def __init__(self, root_dir,split ='train', transform=None, download = None):
        """
        Args:
            root_dir (str): Path to the root directory (e.g., './train' or './test').
            transform (callable, optional): Transformations to apply to the images.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        # Map class names to integer labels
        self.classes = sorted(os.listdir(self.root_dir))  # Sort to maintain order
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Gather all image paths and their corresponding labels
        self.image_paths = []
        self.labels = []
        for cls_name in self.classes:
            cls_folder = os.path.join(self.root_dir, cls_name)
            for img_folder in os.listdir(cls_folder):
                img_names = os.path.join(cls_folder, img_folder)
                for img_name in os.listdir(img_names): 
                    img_path = os.path.join(img_names, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")  # Convert image to RGB

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

class ORCHID(Dataset):
    def __init__(self, root_dir, split ='train', transform=None, download = None):
        """
        Args:
            root_dir (str): Path to the root directory (e.g., './train' or './test').
            transform (callable, optional): Transformations to apply to the images.
        """
            
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform

        # Map class names to integer labels
        self.classes = sorted(os.listdir(self.root_dir))  # Sort to maintain order
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Gather all image paths and their corresponding labels
        self.image_paths = []
        self.labels = []
        for cls_name in self.classes:
            cls_folder = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(cls_folder):
                # img_names = os.path.join(cls_folder, img_folder)
                # for img_name in os.listdir(img_names): 
                img_path = os.path.join(cls_folder, img_name)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")  # Convert image to RGB

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

class QuickGELU(torch.nn.Module):
   def forward(self, x: torch.Tensor):
       return x * torch.sigmoid(1.702 * x)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes, metric):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.metric = metric
        self.clf = LogReg(solver='lbfgs', multi_class='multinomial', warm_start=True)

        print('Logistic regression:')
        print(f'\t solver = L-BFGS')
        print(f"\t classes = {self.num_classes}")
        print(f"\t metric = {self.metric}")

    def set_params(self, d):
        self.clf.set_params(**d)

    @ignore_warnings(category=ConvergenceWarning)
    def fit_logistic_regression(self, X_train, y_train, X_test, y_test):
        if self.metric == 'accuracy':
            self.clf.fit(X_train, y_train)
            test_acc = 100. * self.clf.score(X_test, y_test)
            return test_acc

        elif self.metric == 'mean per-class accuracy':
            self.clf.fit(X_train, y_train)
            pred_test = self.clf.predict(X_test)

            #Get the confusion matrix
            cm = confusion_matrix(y_test, pred_test)
            cm = cm.diagonal() / cm.sum(axis=1)
            test_score = 100. * cm.mean()
            return test_score

        elif self.metric == 'mAP':
            aps_test = []
            for cls in range(self.num_classes):
                self.clf.fit(X_train, y_train[:, cls])
                pred_test = self.clf.decision_function(X_test)
                ap_test = voc_eval_cls(y_test[:, cls], pred_test)
                aps_test.append(ap_test)

            mAP_test = 100. * np.mean(aps_test)
            return mAP_test

        else:
            # rasie not implemented error with message
            raise NotImplementedError(f'Metric {self.metric} not implemented')


class LinearTester():
    def __init__(self, model, train_loader, val_loader, trainval_loader, test_loader, metric,
                 device, num_classes, feature_dim=2048, wd_range=None, max_iter=200,
                 model_name=None, dataset=None, quickgelu=False, multiple=False):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trainval_loader = trainval_loader
        self.test_loader = test_loader
        self.metric = metric
        self.device = device
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.max_iter = max_iter
        self.best_params = {}
        self.model_name = model_name
        self.dataset = dataset
        self.quickgelu = quickgelu
        self.multiple = multiple

        if wd_range is None:
            self.wd_range = torch.logspace(-6, 5, 45)
        else:
            self.wd_range = wd_range

        self.classifier = LogisticRegression(self.feature_dim, self.num_classes, self.metric).to(self.device)

    def get_features(self, train_loader, test_loader, model):
        X_train_feature, y_train = self._inference(train_loader, model, 'train')
        X_test_feature, y_test = self._inference(test_loader, model, 'test')
        return X_train_feature, y_train, X_test_feature, y_test

    def save_features(self):
        X_train_feature, y_train, X_val_feature, y_val = self.get_features(
            self.train_loader, self.val_loader, self.model)
        X_trainval_feature = np.concatenate([X_train_feature, X_val_feature], axis=0)
        y_trainval = np.concatenate([y_train, y_val], axis=0)
        X_test_feature, y_test = self._inference(self.test_loader, self.model, 'test')

        # X_trainval_feature, y_trainval, X_test_feature, y_test = self.get_features(
        #     self.trainval_loader, self.test_loader, self.model)

        base_save_dir = BASE_SAVE_DIR
        if self.quickgelu:
            base_save_dir += '_quickgelu'
        if not os.path.exists(base_save_dir):
            os.makedirs(base_save_dir)
        save_name = f'{self.model_name}_{self.dataset}_features.npz'
        save_name = os.path.join(base_save_dir, save_name)
        np.savez(save_name, X_train_feature=X_train_feature, y_train=y_train,
                 X_val_feature=X_val_feature, y_val=y_val,
                 X_trainval_feature=X_trainval_feature, y_trainval=y_trainval,
                 X_test_feature=X_test_feature, y_test=y_test)
        print(f'Features saved to {save_name}')

    def _inference(self, loader, model, split):
        model.eval()
        feature_vector = []
        labels_vector = []
        with torch.no_grad():
            for data in tqdm(loader, desc=f'Computing features for {split} set'):
                batch_x, batch_y = data
                batch_x = batch_x.to(self.device)
                labels_vector.extend(np.array(batch_y))
                x = model.get_intermediate_layers(batch_x, n=4, return_class_token=True)
                if self.multiple:
                    features = torch.cat([
                        x[0][1],
                        x[1][1],
                        x[2][1],
                        x[3][1],
                        x[3][0].mean(dim=1),
                    ], dim=1)
                    # features = torch.cat([
                    #     x[0][1],
                    #     x[1][1],
                    #     x[2][1],
                    #     x[3][1],
                    # ], dim=1)
                else:
                    features = x[3][1]
                feature_vector.extend(features.cpu().detach().numpy())

        feature_vector = np.array(feature_vector)
        labels_vector = np.array(labels_vector, dtype=int)

        return feature_vector, labels_vector

    def validate(self):
        self.classifier.set_params({'max_iter': self.max_iter})
        X_train_feature, y_train, X_val_feature, y_val = self.get_features(
            self.train_loader, self.val_loader, self.model)
        best_score = 0
        for wd in tqdm(self.wd_range, desc='Selecting best hyperparameters'):
            C = 1. / wd.item()
            self.classifier.set_params({'C': C})
            test_score = self.classifier.fit_logistic_regression(X_train_feature, y_train, X_val_feature, y_val)

            if test_score > best_score:
                best_score = test_score
                self.best_params['C'] = C

    def evaluate(self):
        self.classifier.set_params({'max_iter': self.max_iter})
        print(f"Best hyperparameters {self.best_params}")
        X_trainval_feature, y_trainval, X_test_feature, y_test = self.get_features(
            self.trainval_loader, self.test_loader, self.model
        )
        self.classifier.set_params({'C': self.best_params['C']})
        return self.classifier.fit_logistic_regression(X_trainval_feature, y_trainval, X_test_feature, y_test)


# Data classes and functions
def get_dataset(dset, root, split, transform):
    if dset == t_datasets.ImageFolder:
        return dset(os.path.join(root, split), transform=transform)
    
    else:
        try:
            return dset(root, train=(split == 'train'), transform=transform, download=True)
        except:
            return dset(root, split=split, transform=transform, download=True)


def get_train_valid_loader(dset,
                           data_dir,
                           normalise_dict,
                           batch_size,
                           image_size,
                           random_seed,
                           valid_size=0.2,
                           shuffle=True,
                           num_workers=1,
                           pin_memory=True):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(**normalise_dict)
    print("Train normaliser:", normalize)

    # define transforms
    if dset is t_datasets.MNIST:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    if dset in [t_datasets.Country211]:
        # if we have a predefined validation set
        train_dataset = get_dataset(dset, data_dir, 'train', transform)
        valid_dataset = get_dataset(dset, data_dir, 'valid', transform)
        trainval_dataset = ConcatDataset([train_dataset, valid_dataset])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        trainval_loader = DataLoader(
            trainval_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
    elif dset in [t_datasets.OxfordIIITPet]:
        train_dataset = get_dataset(dset, data_dir, 'trainval', transform)
        valid_dataset = get_dataset(dset, data_dir, 'trainval', transform)
        trainval_dataset = get_dataset(dset, data_dir, 'trainval', transform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        trainval_loader = DataLoader(
            trainval_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )
    else:
        # otherwise we select a random subset of the train set to form the validation set
        train_dataset = get_dataset(dset, data_dir, 'train', transform)
        valid_dataset = get_dataset(dset, data_dir, 'train', transform)
        trainval_dataset = get_dataset(dset, data_dir, 'train', transform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        trainval_loader = DataLoader(
            trainval_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    return train_loader, valid_loader, trainval_loader


def get_test_loader(dset,
                    data_dir,
                    normalise_dict,
                    batch_size,
                    image_size,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=True):
    normalize = transforms.Normalize(**normalise_dict)
    print("Test normaliser:", normalize)

    # define transform
    if dset is t_datasets.MNIST:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            lambda x: x.convert('RGB'),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    dataset = get_dataset(dset, data_dir, 'test', transform)

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def prepare_data(dset, data_dir, batch_size, image_size, normalisation):
    if normalisation:
        normalise_dict = {'mean': [0.48145466, 0.4578275, 0.40821073],
                          'std': [0.26862954, 0.26130258, 0.27577711]}
    else:
        normalise_dict = {'mean': [0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
    train_loader, val_loader, trainval_loader = get_train_valid_loader(
        dset, data_dir, normalise_dict, batch_size, image_size, random_seed=0)
    test_loader = get_test_loader(dset, data_dir, normalise_dict, batch_size, image_size)

    return train_loader, val_loader, trainval_loader, test_loader


class MetricMode(Enum):
    ACC = "accuracy"
    MEAN_CLS_ACC = "mean-class-accuracy"
    MEAN_AP = "mean AP"


# name: {class, root, num_classes, metric}
LINEAR_DATASETS = {
    'aircraft': [t_datasets.FGVCAircraft, f'{DATA_CACHE_DIR}/raw/aircraft', 100, MetricMode.MEAN_CLS_ACC],
    'oral': [ORAL, DATA_DIR, 5, MetricMode.ACC],
    'orchid': [ORCHID, DATA_DIR, 5, MetricMode.ACC],
    'cars': [t_datasets.StanfordCars, f'{DATA_CACHE_DIR}/raw/cars', 196, MetricMode.ACC],
    'cifar10': [t_datasets.CIFAR10, f'{DATA_CACHE_DIR}/cifar10', 10, MetricMode.ACC],
    'cifar100': [t_datasets.CIFAR100, f'{DATA_CACHE_DIR}/cifar100', 100, MetricMode.ACC],
    'dtd': [t_datasets.DTD, f'{DATA_CACHE_DIR}/raw/dtd', 47, MetricMode.ACC],
    'flowers': [t_datasets.Flowers102, f'{DATA_CACHE_DIR}/raw/flowers', 102, MetricMode.MEAN_CLS_ACC],
    'food': [t_datasets.Food101, f'{DATA_CACHE_DIR}/raw/food101', 101, MetricMode.ACC],
    'pets': [t_datasets.OxfordIIITPet, f'{DATA_CACHE_DIR}/raw/pets', 37, MetricMode.MEAN_CLS_ACC],
    'stl10': [t_datasets.STL10, f'{DATA_CACHE_DIR}/stl10', 10, MetricMode.ACC],
    'mnist': [t_datasets.MNIST, f'{DATA_CACHE_DIR}/raw/mnist', 10, MetricMode.ACC],
    'gtsrb': [t_datasets.GTSRB, f'{DATA_CACHE_DIR}/raw/gtsrb', 43, MetricMode.ACC],
    'country211': [t_datasets.Country211, f'{DATA_CACHE_DIR}/raw/country211', 211, MetricMode.ACC],
    'eurosat': [t_datasets.ImageFolder, f'{DATA_CACHE_DIR}/eurosat', 10, MetricMode.ACC],
    'resisc45': [t_datasets.ImageFolder, f'{DATA_CACHE_DIR}/resisc45', 45, MetricMode.ACC],
    'kitti_distance': [t_datasets.ImageFolder, f'{DATA_CACHE_DIR}/kitti_distance', 4, MetricMode.ACC],
}


# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate pretrained self-supervised model via logistic regression.')
    parser.add_argument('--pretrained', type=str, default='/ssd_scratch/karan.p/log/DINOv2_training_ORCHID/checkpoint.pth' )
    parser.add_argument('--multiple', action='store_true', default=False)
    parser.add_argument('-m', '--model', type=str, default='vit_base',
                        help='name of the pretrained model to load and evaluate')
    parser.add_argument('-d', '--dataset', type=str, default='orchid', 
                        help='name of the dataset to evaluate on')
    parser.add_argument('-b', '--batch-size', type=int, default=32, 
                        help='the size of the mini-batches when inferring features')
    parser.add_argument('-i', '--image-size', type=int, default=224, 
                        help='the size of the input images')
    parser.add_argument('-w', '--wd-values', type=int, default=45, 
                        help='the number of weight decay values to validate')
    parser.add_argument('-c', '--C', type=float, default=None, 
                        help='sklearn C value (1 / weight_decay), if not tuning on validation set')
    parser.add_argument('-n', '--no-norm', action='store_true', default=False,
                        help='whether to turn off data normalisation (based on ImageNet values)')
    parser.add_argument('--device', type=str, default='cuda', help='CUDA or CPU training (cuda | cpu)')
    parser.add_argument('--max-iter', type=int, default=500, help='CUDA or CPU training (cuda | cpu)')
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_base_patch16',
                        # help='model architecture: (default: ViT-B/16)')
    parser.add_argument('--quickgelu', action='store_true',
                        help='use quickgelu in all MLP layers')
    args = parser.parse_args()
    args.norm = not args.no_norm
    pprint(args)

    pretrained = args.pretrained
    model_name = args.model

    # check the feature exists or not
    base_save_dir = BASE_SAVE_DIR
    if args.quickgelu:
        base_save_dir += '_quickgelu'
    feature_file_name = f'{args.model}_{args.dataset}_features.npz'
    feature_file_name = os.path.join(base_save_dir, feature_file_name)

    is_feature_exist = False

    feature_time = 0
    if not is_feature_exist:
        # load dataset
        dset, data_dir, num_classes, metric = LINEAR_DATASETS[args.dataset]
        train_loader, val_loader, trainval_loader, test_loader = prepare_data(
            dset, data_dir, args.batch_size, args.image_size, normalisation=args.norm)

        import_student = getattr(models_dinov2, args.model)
        model = import_student(img_size=224,
            patch_size=14,
            init_values=1.0,
            ffn_layer='mlp',
            block_chunks=0,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1)

        checkpoint = torch.load(pretrained, map_location='cpu')

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

        if args.quickgelu:
            print('replacing gelu in {} layers to quickgelu'.format(len(model.blocks)))
            for i in range(len(model.blocks)):
                model.blocks[i].mlp.act = QuickGELU()

        msg = model.load_state_dict(new_state_dict, strict=False)
        print(msg)

        model = model.to(args.device)

        # evaluate model on dataset by fitting logistic regression
        # log running time as well
        start_time = time.time()
        tester = LinearTester(model, train_loader, val_loader, trainval_loader, test_loader,
                              metric, args.device, num_classes, wd_range=torch.logspace(-6, 5, args.wd_values),
                              max_iter=args.max_iter, model_name=args.model, dataset=args.dataset, quickgelu=args.quickgelu, multiple=args.multiple)
        # tester.save_features()
        end_time = time.time()
        feature_time = end_time - start_time

    # ==========linear training part========== #
    start_time = time.time()
    _, _, _, metric = LINEAR_DATASETS[args.dataset]
    data_dict = load_features(feature_file_name, normalize=True)    # apply normalization
    test_acc = lbfgs_gpu(data_dict, metric=metric)

    end_time = time.time()
    running_time = end_time - start_time
    # write to the file
    print(f"Final accuracy for {args.model} on {args.dataset}: {test_acc:.2f}%")

    os.makedirs('{}/{}_quickgelu_{}'.format(RESULT_CACHE_DIR, args.max_iter, args.quickgelu),
                exist_ok=True)
    with open('{}/{}_quickgelu_{}/{}_{}.txt'.format(RESULT_CACHE_DIR,
                                                     args.max_iter,
                                                     args.quickgelu,
                                                     args.model,
                                                     args.dataset),
              'w') as f:
        f.write(str(test_acc))
        f.write('\n')
        f.write('feature extraction time: {}'.format(feature_time))
        f.write('running time: {}'.format(running_time))
