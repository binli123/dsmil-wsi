import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools, glob
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict

import dsmil as mil

def get_bag_feats(csv_file_df, args):
    if args.simclr == 0:
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split(os.sep)[1] + '.csv'
    else:
        feats_csv_path = 'datasets/wsi-tcga-lung/' + os.path.join(csv_file_df.iloc[0].split(os.sep)[-2], csv_file_df.iloc[0].split(os.sep)[-1])
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_class)
    label[int(csv_file_df.iloc[1])] = 1
    return label, feats

def train(train_df, milnet, criterion, optimizer, args):
    csvs = shuffle(train_df).reset_index(drop=True)
    total_loss = 0
    bc = 0
    Tensor = torch.cuda.FloatTensor
    for i in range(len(train_df)):
        optimizer.zero_grad()
        label, feats = get_bag_feats(train_df.iloc[i], args)
        bag_label = Variable(Tensor([label]))
        bag_feats = Variable(Tensor([feats]))
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
        max_prediction, _ = torch.max(ins_prediction, 0)        
        bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
    return total_loss / len(train_df)

def test(test_df, milnet, criterion, optimizer, args):
    csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i in range(len(test_df)):
            label, feats = get_bag_feats(test_df.iloc[i], args)
            bag_label = Variable(Tensor([label]))
            bag_feats = Variable(Tensor([feats]))
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)  
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([label])
            test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_class, pos_label=1)
    for i in range(args.num_class):
        class_prediction_bag = test_predictions[:, i]
        class_prediction_bag[class_prediction_bag>=thresholds_optimal[i]] = 1
        class_prediction_bag[class_prediction_bag<thresholds_optimal[i]] = 0
        test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)
    
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal

def multi_label_roc(labels, predictions, num_class, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    for c in range(0, num_class):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_class', default=2, type=int, help='Number of output classes')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate')
    parser.add_argument('--num_epoch', default=40, type=int, help='Number of total training epochs')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay')
    parser.add_argument('--simclr', default=1, type=int, help='Use newly trained features 1/0(on/off)')
    args = parser.parse_args()
    
    
    i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_class).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_class).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0)
    
    if args.simclr == 0:
        bags_path = pd.read_csv('datasets'+os.sep+'tcga-dataset'+os.sep+'TCGA.csv')
    else:
        luad_list = glob.glob('datasets'+os.sep+'wsi-tcga-lung'+os.sep+'LUAD'+os.sep+'*.csv')
        lusc_list = glob.glob('datasets'+os.sep+'wsi-tcga-lung'+os.sep+'LUSC'+os.sep+'*.csv')
        luad_df = pd.DataFrame(luad_list)
        luad_df['label'] = 0
        luad_df.to_csv('datasets/wsi-tcga-lung/LUAD.csv', index=False)        
        lusc_df = pd.DataFrame(lusc_list)
        lusc_df['label'] = 1
        lusc_df.to_csv('datasets/wsi-tcga-lung/LUSC.csv', index=False)        
        bags_path = luad_df.append(lusc_df, ignore_index=True)
        bags_path = shuffle(bags_path)
        bags_path.to_csv('datasets/wsi-tcga-lung/TCGA.csv', index=False)
    train_path = bags_path.iloc[0:int(len(bags_path)*0.8), :]
    test_path = bags_path.iloc[int(len(bags_path)*0.8):, :]
    
    for epoch in range(1, args.num_epoch):
        train_path = shuffle(train_path).reset_index(drop=True)
        test_path = shuffle(test_path).reset_index(drop=True)
        train_loss_bag = train(train_path, milnet, criterion, optimizer, args) # iterate all bags
        test_loss_bag, avg_score, aucs, thresholds_optimal = test(test_path, milnet, criterion, optimizer, args)
        print('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, average score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
              (epoch, args.num_epoch, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
        scheduler.step()

if __name__ == '__main__':
    main()