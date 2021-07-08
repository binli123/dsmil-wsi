import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, copy, itertools
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support
from sklearn.datasets import load_svmlight_file
from collections import OrderedDict


def get_data(file_path):
    df = pd.read_csv(file_path)
    df = pd.DataFrame(df)
    df = df[df.columns[0]]
    data_list = []    
    for i in range(0, df.shape[0]):  
        data = str(df.iloc[i]).split(' ')
        ids = data[0].split(':')
        idi = int(ids[0])
        idb = int(ids[1])
        idc = int(ids[2])
        data = data[1:]
        feature_vector = np.zeros(len(data))  
        for i, feature in enumerate(data):
            feature_data = feature.split(':')
            if len(feature_data) == 2:
                feature_vector[i] = feature_data[1]
        data_list.append([idi, idb, idc, feature_vector])
    return data_list

def get_bag(data, idb):
    data_array = np.array(data, dtype=object)
    bag_id = data_array[:, 1]
    return data_array[np.where(bag_id == idb)]

def epoch_train(bag_ins_list, optimizer, criterion, milnet, args):
    epoch_loss = 0
    for i, data in enumerate(bag_ins_list):
        optimizer.zero_grad()
        data_bag_list = shuffle(data[1])
        data_tensor = torch.from_numpy(np.stack(data_bag_list)).float().cuda()
        data_tensor = data_tensor[:, 0:args.num_feats]
        label = torch.from_numpy(np.array(int(np.clip(data[0], 0, 1)))).float().cuda()
        classes, bag_prediction, _, _ = milnet(data_tensor) # n X L
        max_prediction, index = torch.max(classes, 0)
        loss_bag = criterion(bag_prediction.view(1, -1), label.view(1, -1))
        loss_max = criterion(max_prediction.view(1, -1), label.view(1, -1))
        loss_total = 0.5*loss_bag + 0.5*loss_max
        loss_total = loss_total.mean()
        loss_total.backward()
        optimizer.step()  
        epoch_loss = epoch_loss + loss_total.item()
    return epoch_loss / len(bag_ins_list)

def epoch_test(bag_ins_list, criterion, milnet, args):
    bag_labels = []
    bag_predictions = []
    epoch_loss = 0
    with torch.no_grad():
        for i, data in enumerate(bag_ins_list):
            bag_labels.append(np.clip(data[0], 0, 1))
            data_tensor = torch.from_numpy(np.stack(data[1])).float().cuda()
            data_tensor = data_tensor[:, 0:args.num_feats]
            label = torch.from_numpy(np.array(int(np.clip(data[0], 0, 1)))).float().cuda()
            classes, bag_prediction, _, _ = milnet(data_tensor) # n X L
            max_prediction, index = torch.max(classes, 0)
            loss_bag = criterion(bag_prediction.view(1, -1), label.view(1, -1))
            loss_max = criterion(max_prediction.view(1, -1), label.view(1, -1))
            loss_total = 0.5*loss_bag + 0.5*loss_max
            loss_total = loss_total.mean()
            bag_predictions.append(torch.sigmoid(bag_prediction).cpu().squeeze().numpy())
            epoch_loss = epoch_loss + loss_total.item()
    epoch_loss = epoch_loss / len(bag_ins_list)
    return epoch_loss, bag_labels, bag_predictions

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def five_scores(bag_labels, bag_predictions):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore

def cross_validation_set(in_list, fold, index):
    csv_list = copy.deepcopy(in_list)
    n = int(len(csv_list)/fold)
    chunked = [csv_list[i:i+n] for i in range(0, len(csv_list), n)]
    test_list = chunked.pop(index)
    return list(itertools.chain.from_iterable(chunked)), test_list

def compute_pos_weight(bags_list):
    pos_count = 0
    for item in bags_list:
        pos_count = pos_count + np.clip(item[0], 0, 1)
    return (len(bags_list)-pos_count)/pos_count

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on classfical MIL datasets')
    parser.add_argument('--datasets', default='musk1', type=str, help='Choose MIL datasets from: musk1, musk2, elephant, fox, tiger [musk1]')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epoch', default=40, type=int, help='Number of total training epochs [40]')
    parser.add_argument('--cv_fold', default=10, type=int, help='Number of cross validation fold [10]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    args = parser.parse_args()
    
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil
    
    if args.datasets == 'musk1':
        data_all = get_data('datasets/mil_dataset/Musk/musk1norm.svm')
        args.num_feats = 166
    if args.datasets == 'musk2':
        data_all = get_data('datasets/mil_dataset/Musk/musk2norm.svm')
        args.num_feats = 166
    if args.datasets == 'elephant':
        data_all = get_data('datasets/mil_dataset/Elephant/data_100x100.svm')
        args.num_feats = 230
    if args.datasets == 'fox':
        data_all = get_data('datasets/mil_dataset/Fox/data_100x100.svm')
        args.num_feats = 230
    if args.datasets == 'tiger':
        data_all = get_data('datasets/mil_dataset/Tiger/data_100x100.svm')
        args.num_feats = 230  
    
    bag_ins_list = []
    num_bag = data_all[-1][1]+1
    for i in range(num_bag):
        bag_data = get_bag(data_all, i)
        bag_label = bag_data[0, 2]
        bag_vector = bag_data[:, 3]
        bag_ins_list.append([bag_label, bag_vector])
    bag_ins_list = shuffle(bag_ins_list)
    
    ### check both classes exist in testing bags
    valid_bags = 1
    while(valid_bags):
        bag_ins_list = shuffle(bag_ins_list)
        for k in range (0, args.cv_fold):
            bags_list, test_list = cross_validation_set(bag_ins_list, fold=args.cv_fold, index=k)
            bag_labels = 0
            for i, data in enumerate(test_list):
                bag_labels = np.clip(data[0], 0, 1) + bag_labels
            if bag_labels > 0:
                valid_bags = 0         
    
    acs = []
    print('Dataset: ' + args.datasets)
    for k in range(0, args.cv_fold):
        print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
        bags_list, test_list = cross_validation_set(bag_ins_list, fold=args.cv_fold, index=k)
        i_classifier = mil.FCLayer(args.num_feats, 1)
        b_classifier = mil.BClassifier(input_size=args.num_feats, output_class=1)
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        pos_weight = torch.tensor(compute_pos_weight(bags_list))
        criterion = nn.BCEWithLogitsLoss(pos_weight)
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0)
        optimal_ac = 0
        for epoch in range(0, args.num_epoch):
            train_loss = epoch_train(bags_list, optimizer, criterion, milnet, args) # iterate all bags
            test_loss, bag_labels, bag_predictions = epoch_test(test_list, criterion, milnet, args)
            accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_predictions)
            sys.stdout.write('\r Epoch [%d/%d] train loss: %.4f, test loss: %.4f, accuracy: %.4f, aug score: %.4f, precision: %.4f, recall: %.4f, fscore: %.4f ' % 
                  (epoch+1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore))
            optimal_ac = max(accuracy, optimal_ac)
            scheduler.step()
        print('\n Optimal accuracy: %.4f ' % (optimal_ac))
        acs.append(optimal_ac)
    print('Cross validation accuracy mean: %.4f, std %.4f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
    
    
if __name__ == '__main__':
    main()