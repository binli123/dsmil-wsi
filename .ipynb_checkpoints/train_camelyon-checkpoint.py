import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as VF
from torchvision import transforms, utils

import sys, os, argparse, glob
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from collections import OrderedDict

import dsmil as mil

def get_bag_feats(csv_file_path):
    df = pd.read_csv(csv_file_path)
    feats = df.iloc[:, 1:]
    feats = shuffle(feats)
    feats = feats.to_numpy()
    label = df.iloc[0, 0]
    return label, feats

def train(csv_path, milnet, criterion, optimizer, args):
    Tensor = torch.cuda.FloatTensor
    csvs = glob.glob(csv_path+'/*.csv')
    csvs = shuffle(csvs)
    total_loss = 0
    batch_size = 16
    bc = 0
    for csv in csvs:
        bc = bc+1
        optimizer.zero_grad()
        label, feats = get_bag_feats(csv)
        bag_label = Variable(Tensor([label]))
        bag_feats = Variable(Tensor([feats]))
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        max_prediction = torch.max(ins_prediction)   
        bag_loss = criterion(bag_prediction.view(1, 1), bag_label.view(1, 1))
        max_loss = criterion(max_prediction.view(1, 1), bag_label.view(1, 1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Bag [%d/%d] bag loss: %.4f' % (bc, len(csvs), loss.item()))
    return total_loss / len(csvs)

def test(csv_path, milnet, criterion, optimizer, args):
    Tensor = torch.cuda.FloatTensor
    csvs = glob.glob(csv_path+'/*.csv')
    bc = 0
    total_loss = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for csv in csvs:
            label, feats = get_bag_feats(csv)
            bag_label = Variable(Tensor([label]))
            bag_feats = Variable(Tensor([feats]))
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction = torch.max(ins_prediction)
            bag_loss = criterion(bag_prediction.view(1, 1), bag_label.view(1, 1))
            max_loss = criterion(max_prediction.view(1, 1), bag_label.view(1, 1))
            loss = 0.0*bag_loss + 1.0*max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Bag [%d/%d] bag loss: %.4f' % (bc, len(csvs), loss.item()))
            bc = bc+1
            test_labels.extend([label])
            test_predictions.extend([(0.5*torch.sigmoid(max_prediction)+0.5*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    fpr, tpr, thresholds = roc_curve(test_labels, test_predictions)   
    _, _, optimal_threshold = optimal_thresh(fpr, tpr, thresholds)
    auc = roc_auc_score(test_labels, test_predictions)
    test_predictions[test_predictions>=optimal_threshold] = 1
    test_predictions[test_predictions<optimal_threshold] = 0
    accuracy = accuracy_score(test_labels, test_predictions)  
    
    return total_loss / len(csvs), accuracy, auc

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_class', default=1, type=int, help='Number of output classes')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate')
    parser.add_argument('--num_epoch', default=100, type=int, help='Number of total training epochs')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay')
    args = parser.parse_args()
    
    
    i_classifier = mil.FCLayer(args.feats_size, args.num_class)
    b_classifier = mil.BClassifier(args.feats_size, args.num_class)
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0)
    
    train_csvs = 'datasets/camelyon_data_feats/train'
    test_csvs = 'datasets/camelyon_data_feats/test'
    for epoch in range(args.num_epoch):
        milnet.train()
        train_loss = train(train_csvs, milnet, criterion, optimizer, args)
        milnet.eval()
        test_loss, accuracy, auc = test(test_csvs, milnet, criterion, optimizer, args)
        scheduler.step()
        print('\r Epoch [%d/%d] train loss: %.4f, test loss: %.4f, accuracy: %.4f, auc: %.4f' % (epoch, args.num_epoch, train_loss, test_loss, accuracy, auc))
        
if __name__ == '__main__':
    main()