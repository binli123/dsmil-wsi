import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms
import matplotlib.pyplot as plt
import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score, accuracy_score, hamming_loss
from sklearn.model_selection import KFold, StratifiedKFold
from collections import OrderedDict
import json
from tqdm import tqdm
import torch.nn.functional as F
import csv
from sklearn.metrics import confusion_matrix

os.chdir("/ssd_scratch/karan.p")

def get_bag_feats(csv_file_df, args):
    if args.dataset == 'TCGA-lung-default':
        feats_csv_path = 'datasets/tcga-dataset/tcga_lung_data_feats/' + csv_file_df.iloc[0].split('/')[1] + '.csv'
    else:
        feats_csv_path = csv_file_df.iloc[0]
    df = pd.read_csv(feats_csv_path)
    feats = shuffle(df).reset_index(drop=True)
    feats = feats.to_numpy()
    label = np.zeros(args.num_classes)
    if args.num_classes==1:
        label[0] = csv_file_df.iloc[1]
    else:
        if int(csv_file_df.iloc[1])<=(len(label)-1):
            label[int(csv_file_df.iloc[1])] = 1
        
    return label, feats, feats_csv_path

def generate_pt_files(args, df):
    temp_train_dir = "temp_train"
    if os.path.exists(temp_train_dir):
        import shutil
        shutil.rmtree(temp_train_dir, ignore_errors=True)
    os.makedirs(temp_train_dir, exist_ok=True)
    print('Creating intermediate training files.')
    for i in tqdm(range(len(df))):
        label, feats, feats_csv_path = get_bag_feats(df.iloc[i], args)
        bag_label = torch.tensor(np.array([label]), dtype=torch.float32)
        bag_feats = torch.tensor(np.array(feats), dtype=torch.float32)
        repeated_label = bag_label.repeat(bag_feats.size(0), 1)
        stacked_data = torch.cat((bag_feats, repeated_label), dim=1)
        # Save the stacked data into a .pt file
        pt_file_path = os.path.join(temp_train_dir, os.path.splitext(feats_csv_path)[0].split(os.sep)[-1] + ".pt")
        torch.save(stacked_data, pt_file_path)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Focal Loss implementation
        
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): Focusing parameter. Defaults to 2.
            reduction (str, optional): 'mean', 'sum' or 'none'. Defaults to 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-7

    def forward(self, inputs, targets):
        """
        Calculate focal loss
        
        Args:
            inputs (Tensor): Model predictions (N, C) or (N,)
            targets (Tensor): Target values (N, C) or (N,)
            
        Returns:
            Tensor: Computed focal loss
        """
        # Convert targets to float for calculations
        targets = targets.float()
        
        # Get sigmoid of inputs
        probs = torch.sigmoid(inputs)
        
        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def train(args, train_df, milnet, criterion, optimizer, class_weights):
    milnet.train()
    dirs = shuffle(train_df)
    total_loss = 0
    train_labels = []
    train_predictions = []
    Tensor = torch.cuda.FloatTensor
    for i, item in enumerate(dirs):
        slide_id = os.path.basename(item)
        cluster_tensor = args.slide_cluster_dict[slide_id[:-3]]
        optimizer.zero_grad()
        stacked_data = torch.load(item, map_location='cuda:0')
        bag_label = Tensor(stacked_data[0, args.feats_size:]).unsqueeze(0)
        bag_feats = Tensor(stacked_data[:, :args.feats_size])
        bag_feats = dropout_patches(bag_feats, 1-args.dropout_patch)
        bag_feats = bag_feats.view(-1, args.feats_size)
        ins_prediction, bag_prediction, _, _ = milnet(bag_feats, cluster_tensor)
        max_prediction, _ = torch.max(ins_prediction, 0)    

        if args.loss_func == "focal_loss":
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))

        elif class_weights is not None:
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            bag_loss = (bag_loss * class_weights[bag_label.long()]).mean()
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = (max_loss * class_weights[bag_label.long()]).mean()

        else:
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))

        train_labels.extend([bag_label.squeeze().cpu().numpy().astype(int)])
        
        if args.average:
                train_predictions.extend([(torch.sigmoid(max_prediction)+torch.sigmoid(bag_prediction)).squeeze().cpu().detach().numpy()])
        else: train_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().detach().numpy()])

    train_labels = np.array(train_labels)
    train_predictions = np.array(train_predictions)
    auc_value, _, thresholds_optimal, train_auc = multi_label_roc(train_labels, train_predictions, args.num_classes, pos_label=1)
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(train_predictions)
        class_prediction_bag[train_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[train_predictions<thresholds_optimal[0]] = 0
        train_predictions = class_prediction_bag
        train_labels = np.squeeze(train_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(train_predictions[:, i])
            class_prediction_bag[train_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[train_predictions[:, i]<thresholds_optimal[i]] = 0
            train_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(train_df)):
        bag_score = np.array_equal(train_labels[i], train_predictions[i]) + bag_score         
    train_acc = bag_score / len(train_df)

    return total_loss / len(train_df), train_acc, train_auc

def dropout_patches(feats, p):
    num_rows = feats.size(0)
    num_rows_to_select = int(num_rows * p)
    random_indices = torch.randperm(num_rows)[:num_rows_to_select]
    selected_rows = feats[random_indices]
    return selected_rows


def roc_threshold(label, prediction):

    # Handle binary classification
    if prediction.ndim == 1 or prediction.shape[1] == 2:
        # Ensure prediction is a 1D array
        if prediction.ndim > 1:
            prediction = prediction[:, 1]
            prediction = prediction.ravel()

        fpr, tpr, thresholds = roc_curve(label, prediction, pos_label=1)
        # Find the optimal threshold (You can customize the optimal_thresh function)
        optimal_idx = (tpr - fpr).argmax()
        threshold_optimal = thresholds[optimal_idx]
        c_auc = roc_auc_score(label, prediction)
        return c_auc, threshold_optimal

    # Handle multi-class classification
    elif prediction.shape[1] >2:
        c_auc = roc_auc_score(label, prediction, multi_class='ovr')
        return c_auc, None  # Thresholds are not defined for multi-class AUC

    else:
        raise ValueError("Invalid input shapes for label or prediction.")

import numpy as np
from sklearn.metrics import confusion_matrix

def eval_metric_classwise(oprob, label, num_classes):
    """
    Compute class-wise accuracy, precision, recall, and F1 score.

    Args:
        oprob (np.ndarray): Output probabilities (N, num_classes).
        label (np.ndarray): Ground-truth labels (N,).
        num_classes (int): Number of classes.

    Returns:
        metrics (dict): Dictionary containing class-wise metrics.
    """
    # Ensure inputs are numpy arrays
    oprob = np.asarray(oprob, dtype=np.float32)
    label = np.asarray(label, dtype=np.int64)
    
    metrics = {}
    # oprob_ = oprob[:, 1] if oprob.shape[1] > 1 else oprob.squeeze()
    
    # _, threshold = roc_threshold(label, oprob)
    if oprob.ndim == 1 or oprob.shape[1] == 2:
        # pred = oprob_ > 0.5
        cm = confusion_matrix(label, oprob)
    else:
        pred = np.argmax(oprob, axis=1)
    
    for cls in range(num_classes+1):
        cls_mask = label == cls  # Mask for current class
        pred_cls = oprob == cls
        # prob_cls = oprob[:, cls] if oprob.ndim > 1 else oprob
        
        # Compute metrics for the current class
        TP = np.sum(pred_cls & cls_mask).astype(np.float32)
        TN = np.sum(~pred_cls & ~cls_mask).astype(np.float32)
        FP = np.sum(pred_cls & ~cls_mask).astype(np.float32)
        FN = np.sum(~pred_cls & cls_mask).astype(np.float32)
        
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-12)
        precision = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)
        F1 = 2 * (precision * recall) / (precision + recall + 1e-12)
        
        # Store metrics for the current class
        metrics[f"class_{cls}"] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "F1": float(F1),
            "cm": cm if cls == 0 else None  # Only store cm once (avoid redundancy)
        }
    
    return metrics

# def eval_metric_classwise(oprob, label, num_classes):
#     """
#     Compute class-wise accuracy, precision, recall, F1 score, and AUC.

#     Args:
#         oprob (torch.Tensor): Output probabilities (N, num_classes).
#         label (torch.Tensor): Ground-truth labels (N,).
#         num_classes (int): Number of classes.

#     Returns:
#         metrics (dict): Dictionary containing class-wise metrics.
#     """
#     metrics = {}
#     label = torch.tensor(label, device= "cpu")
#     np_array = np.array(oprob, dtype=np.float32)

#     # Convert to tensor
#     oprob = torch.from_numpy(np_array)
#     # oprob = torch.tensor(oprob, device= "cpu")
#     # oprob_ = oprob[:, 1]

#     auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
#     if oprob.ndim == 1 or oprob.shape[1] == 2:
#         pred = oprob > threshold
#         cm = confusion_matrix(label.cpu().numpy(), pred.cpu().numpy())
#     else: 
#         pred = np.argmax(oprob.detach().cpu().numpy(), axis=1)
#     for cls in range(num_classes+1):
#         cls_mask = label == cls  # Mask for current class
#         pred_cls = pred == cls
#         # prob_cls = oprob[:, cls]

#         pred_cls =  torch.tensor(pred_cls, device= 'cpu')
#         # Compute metrics for the current class
#         TP = (pred_cls & cls_mask).sum().float()
#         TN = ((~pred_cls) & (~cls_mask)).sum().float()
#         FP = (pred_cls & (~cls_mask)).sum().float()
#         FN = ((~pred_cls) & cls_mask).sum().float()

#         accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-12)
#         precision = TP / (TP + FP + 1e-12)
#         recall = TP / (TP + FN + 1e-12)
#         F1 = 2 * (precision * recall) / (precision + recall + 1e-12)

#         # Compute AUC using sklearn for the current class
#         auc = roc_auc_score(cls_mask.cpu().numpy(), oprob.detach().cpu().numpy()) if cls_mask.sum() > 0 else 0.0

#         # Store metrics for the current class
#         metrics[f"class_{cls}"] = {
#             "accuracy": accuracy.item(),
#             "precision": precision.item(),
#             "recall": recall.item(),
#             "F1": F1.item(),
#             "AUC": auc,
#             "cm": cm
#         }

#     return metrics


def test(args, test_df, milnet, criterion, thresholds=None, return_predictions=False, class_weights= None):
    milnet.eval()
    total_loss = 0
    test_labels = []
    test_predictions = []
    features_list = []
    labels_list = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():
        for i, item in enumerate(test_df):
            slide_id = os.path.basename(item)
            cluster_tensor = args.slide_cluster_dict[slide_id[:-3]]
            stacked_data = torch.load(item, map_location='cuda:0')
            bag_label = Tensor(stacked_data[0, args.feats_size:]).unsqueeze(0)
            bag_feats = Tensor(stacked_data[:, :args.feats_size])
            bag_feats = dropout_patches(bag_feats, 1-args.dropout_patch)
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, slide_features = milnet(bag_feats,cluster_tensor)

            features_list.append(slide_features.cpu().detach())
            labels_list.append(int(bag_label.item()))

            max_prediction, _ = torch.max(ins_prediction, 0)

            if args.loss_func == "focal_loss":
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
  
            elif class_weights is not None:
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                bag_loss = (bag_loss * class_weights[bag_label.long()]).mean()
                max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
                max_loss = (max_loss * class_weights[bag_label.long()]).mean()

            else:
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
                max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
            
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_df), loss.item()))
            test_labels.extend([bag_label.squeeze().cpu().numpy().astype(int)])
            if args.average:
                test_predictions.extend([(torch.sigmoid(max_prediction)+torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
            else: test_predictions.extend([torch.sigmoid(bag_prediction).squeeze().cpu().numpy()])
    if args.train == False:
        save_dict = {'features': features_list, 'labels': labels_list}
        torch.save(save_dict, args.test_model_path+'/slide_features_labels.pt')

    test_labels = np.array(test_labels)
    test_predictions_ = test_predictions
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal, test_auc = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if thresholds: thresholds_optimal = thresholds
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_df)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_df)

    classwise_metrics = eval_metric_classwise(test_predictions, test_labels, args.num_classes)
    
    if return_predictions:
        return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, test_predictions, test_labels
    return total_loss / len(test_df), avg_score, auc_value, thresholds_optimal, test_auc, classwise_metrics

def multi_label_roc(labels, predictions, num_classes, pos_label=1, average='macro'):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    if labels.ndim == 1:
        labels = np.expand_dims(labels, axis=-1)
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        # c_auc = roc_auc_score(label, prediction)
        try:
            c_auc = roc_auc_score(label, prediction)
            print("ROC AUC score:", c_auc)
        except ValueError as e:
            if str(e) == "Only one class present in y_true. ROC AUC score is not defined in that case.":
                print("ROC AUC score is not defined when only one class is present in y_true. c_auc is set to 1.")
                c_auc = 1
            else:
                raise e

        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)

    # Compute the overall AUC score
    try:
        overall_auc = roc_auc_score(labels, predictions, average=average)
        print(f"Overall {average} ROC AUC score:", overall_auc)
    except ValueError as e:
        if str(e) == "Only one class present in y_true. ROC AUC score is not defined in that case.":
            print("Overall ROC AUC score is not defined when only one class is present in y_true. Setting overall_auc to 1.")
            overall_auc = 1
        else:
            raise e
    return aucs, thresholds, thresholds_optimal, overall_auc

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def print_epoch_info(epoch, args, train_loss_bag, train_acc, train_auc, test_loss_bag, avg_score, aucs, test_auc):
    if args.dataset.startswith('TCGA-lung'):
        print('\r Epoch [%d/%d] train loss: %.4f val loss: %.4f, val acc score: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f' % 
                (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, aucs[0], aucs[1]))
    else:
        print('\r Epoch [%d/%d] train loss: %.4f train acc: %.4f val_auc: %.4f val loss: %.4f val acc score: %.4f valauc: %.4f ,AUC: %s' % 
    (epoch, args.num_epochs, train_loss_bag, train_acc, train_auc, test_loss_bag, avg_score, test_auc,
     ' | '.join('class-{}>>{:.4f}'.format(i, auc) for i, auc in enumerate(aucs))))
        
def get_current_score(avg_score, aucs):
    current_score = (sum(aucs) + avg_score)/2
    return current_score

def save_model(args, fold, run, save_path, model, thresholds_optimal, epoch_data):
    # Construct the filename including the fold number
    save_name = os.path.join(save_path, f'fold_{fold}_{run+1}.pth')
    torch.save(model.state_dict(), save_name)
    print_save_message(args, save_name, thresholds_optimal)

    data_to_save = {
        'thresholds_optimal': [float(x) for x in thresholds_optimal],
        'epoch_data': {
            'epoch': epoch_data[0],
            'train_loss_bag': epoch_data[2],
            'train_acc': epoch_data[3],
            'train_auc': epoch_data[4],
            'val_loss_bag': epoch_data[5],
            'val_acc': epoch_data[6],
            'val_auc': epoch_data[8],
            **{f'val_auc_{i+1}': epoch_data[7][i] for i in range(args.num_classes)}
        }
    }
    file_name = os.path.join(save_path, f'fold_{fold}_{run+1}.json')
    with open(file_name, 'w') as f:
        json.dump(data_to_save, f, indent=4)

def print_save_message(args, save_name, thresholds_optimal):
    if args.dataset.startswith('TCGA-lung'):
        print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
    else:
        print('Best model saved at: ' + save_name)
        print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))

def reOrganize_mDATA(dataset_csv, fold_csv, set_type, label_name='label'):

    SlideNames = []
    Label = []

    mDATA_slides = pd.read_csv(fold_csv)
    mDATA_label = pd.read_csv(dataset_csv)

    temp_SlideNames = mDATA_slides[set_type]

    mDATA = mDATA_label[mDATA_label['slide_id'].isin(temp_SlideNames)]

    mapping = {'Stage_1':0, 'Stage_2':1, 'Stage_3':2}
    mDATA = mDATA.replace({label_name: mapping})
    
    SlideNames = mDATA['slide_id'].tolist()
    Label = mDATA[label_name].tolist()
    try:
        quality = mDATA_label['quality']
    except Exception as e:
        quality = np.ones(len(mDATA), dtype=np.int8)
    ## to test
    return SlideNames, Label, quality

def write_test_metrics(test_classwise,iteration):
    # Extract class-wise metrics, excluding confusion matrix
    class_metrics = {}
    for class_name, metrics in test_classwise.items():
        class_metrics.update({f"{class_name}_{k}": v for k, v in metrics.items() if k != "cm"})

        # Save confusion matrix as PNG
        cm = metrics["cm"]
        if cm is None:
            continue
        plt.figure(figsize=(4, 4))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        plt.xticks([0, 1], ["A", "S"])
        plt.yticks([0, 1], ["A", "S"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {class_name}")

        # Annotate the heatmap
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

        cm_filename = os.path.join('/home/karan.padariya/dsmil-wsi/results/'+ f"{iteration}_confusion_matrix_test.png")
        plt.savefig(cm_filename)
        plt.close()
        
    # Convert to DataFrame
    df = pd.DataFrame([class_metrics])

    # Write to CSV (append if file exists, else create)
    file_path = os.path.join('/home/karan.padariya/dsmil-wsi/results/', "metrics_test.csv")
    if os.path.exists(file_path):
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, mode='w', header=True, index=False)

def read_test_data(file_path, label_name='label'):
    mapping = {'ADENO': 0, 'SQUAMOUS': 1, 'Stage_3__': 2}
    
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Remove rows where quality == 0
    df = df[df['quality'] != 0]
    
    # Map label values
    df[label_name] = df[label_name].map(mapping)
    
    # Extract lists
    slide_ids = df['slide_id'].tolist()
    labels = df[label_name].tolist()
    qualities = df['quality'].tolist()
    
    return slide_ids, labels, qualities

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=1024, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--lr', default=0.0001, type=float, help='Initial learning rate [0.0001]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [100]')
    parser.add_argument('--stop_epochs', default=50, type=int, help='Skip remaining epochs if training has not improved after N epochs [10]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,1,2,4), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight decay [1e-3]')
    parser.add_argument('--dataset', default='lung_tcga_tumor', type=str, help='Dataset folder name')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--average', type=bool, default=False, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--eval_scheme', default='10-time-train+valid+test', type=str, help='Evaluation scheme [10-fold-cv | 10-fold-cv-standalone-test | 10-time-train+valid+test ]')
    parser.add_argument('--dataset_csv', default="/home/karan.padariya/CLAM/dataset_csv/updated_tcga-LUAD&LUSC_updated_modified.csv", type=str)  ## Dataset_csv
    parser.add_argument('--leiden_ft', default="/home/karan.padariya/CLAM/lieden_dist", type = str)
    parser.add_argument('--splits_dir', default='/home/karan.padariya/CLAM/splits/task_1_tumor_vs_normal_100', type=str)  ## Dataset_csv
    parser.add_argument('--loss_func',default='BCEW',type=str,choices=['BCE', 'BCEW', 'focal_loss'])
    parser.add_argument('--focal_gamma', type=float, default=4.0, help='Focusing parameter gamma for Focal Loss')
    parser.add_argument('--train', type=bool, default=True, help="train or inferance")
    parser.add_argument('--test_model_path', type=str, default="/ssd_scratch/karan.p/weights/20250411")
    # Processing TCGA-44-7661-01Z-00-DX1
    args = parser.parse_args()
    print(args.eval_scheme)

    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil

    def apply_sparse_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def calculate_class_weights(labels):
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(class_weights)

    def init_model(args):
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, num_cluster= args.num_clusters, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
        milnet.apply(lambda m: apply_sparse_init(m))

        # Calculate alpha (optional class weights) for Focal Loss
        if args.dataset == 'TCGA-lung-default':
            # Example for binary classification
            alpha = torch.tensor([0.5, 0.5]).cuda()  # You can adjust these weights based on class distribution
        else:
            # For multi-class, calculate based on class distribution
            all_labels = []
            with open(os.path.join('datasets', args.dataset, args.dataset+'.csv'), 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for row in reader:
                    all_labels.append(int(row[1]))  # assuming label is in second column
            
        class_counts = np.bincount(all_labels)
        total_samples = len(all_labels)
        alpha = torch.tensor([total_samples/(len(class_counts)*count) for count in class_counts]).cuda()
        alpha = F.softmax(alpha, dim=0) 

        if args.loss_func =="BCE":
            criterion = nn.BCEWithLogitsLoss()
        elif args.loss_func =="BCEW":
            criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif args.loss_func == "focal_loss":
            criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma).cuda()

        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
        return milnet, criterion, optimizer, scheduler
    
    if args.dataset == 'TCGA-lung-default':
        bags_csv = 'datasets/tcga-dataset/TCGA.csv'
    else:
        bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')

    df_bags = pd.read_csv(bags_csv)

    extracted_ids = []
    labels = []

    # Loop through each row and extract the values
    for index, row in df_bags.iterrows():
        # Extract the specific part from the "0" column and append it
        extracted_id = row["0"].split('/')[-1].split('.')[0]
        extracted_ids.append(extracted_id)
        
        # Append the label value as-is
        labels.append(row["label"])

    class_weights = calculate_class_weights(labels).cuda()
    # generate_pt_files(args, df_bags)
 
    if args.eval_scheme == '10-fold-cv':
        bags_path = glob.glob('temp_train/*.pt')
        # bags_path = bags_path.sample(n=200)
        # kf = KFold(n_splits=5, shuffle=True, random_state=42)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        skf.get_n_splits(extracted_ids, labels)
        fold_results = []

        save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))

        for fold, (train_index, test_index) in enumerate(skf.split(extracted_ids, labels)):
            print(f"Starting CV fold {fold}.")
            milnet, criterion, optimizer, scheduler = init_model(args)
            train_path = ["temp_train/" +extracted_ids[i]+".pt" for i in train_index]
            test_path = ["temp_train/" +extracted_ids[i]+".pt"  for i in test_index]
            fold_best_score = 0
            best_ac = 0
            best_auc = 0
            counter = 0

            for epoch in range(1, args.num_epochs+1):
                counter += 1
                train_loss_bag, train_acc, train_auc = train(args, train_path, milnet, criterion, optimizer, class_weights) # iterate all bags
                test_loss_bag, avg_score, aucs, thresholds_optimal, test_auc = test(args, test_path, milnet, criterion,class_weights)
                
                print_epoch_info(epoch, args, train_loss_bag, train_acc, train_auc, test_loss_bag, avg_score, aucs, test_auc)
                epoch_data = (epoch, args, train_loss_bag, train_acc, train_auc, test_loss_bag, avg_score, aucs, test_auc)
                scheduler.step()

                current_score = get_current_score(avg_score, aucs)
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_ac = avg_score
                    best_auc = aucs
                    save_model(args, fold, run, save_path, milnet, thresholds_optimal, epoch_data)
                if counter > args.stop_epochs: break
            fold_results.append((best_ac, best_auc))
        mean_ac = np.mean(np.array([i[0] for i in fold_results]))
        mean_auc = np.mean(np.array([i[1] for i in fold_results]), axis=0)
        # Print mean and std deviation for each class
        print(f"Final results: Mean Accuracy: {mean_ac}")
        for i, mean_score in enumerate(mean_auc):
            print(f"Class {i}: Mean AUC = {mean_score:.4f}")


    elif args.eval_scheme == '10-time-train+valid+test':
        bags_path = glob.glob('temp_train/*.pt')
        # bags_path = bags_path.sample(n=50, random_state=42)
        fold_results = []

        save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))

        for iteration in range(10):
            print(f"Starting iteration {iteration + 1}.")
            try:
                path = os.path.join(args.leiden_ft,f"cluster_tcga_distribution_{iteration}.csv")
                args.leiden_features = pd.read_csv(path)
                args.cluster_columns = [col for col in args.leiden_features.columns if col.startswith("cluster_")]
                args.num_clusters = len(args.cluster_columns)
                args.slide_cluster_dict = {row["slide_id"].split(".")[0]: torch.tensor(row[args.cluster_columns].values.astype(float))
                                      for _, row in args.leiden_features.iterrows()
                                      }
            except Exception as e:
                print(e)
                args.num_clusters = 0

            milnet, criterion, optimizer, scheduler = init_model(args)

            # bags_path = shuffle(bags_path)
            # total_samples = len(bags_path)
            # train_end = int(total_samples * (1-args.split-0.1))
            # val_end = train_end + int(total_samples * 0.1)

            fold_csv = os.path.join(args.splits_dir, 'splits_'+ str(iteration) + '.csv')

            SlideNames_train, _, _ = reOrganize_mDATA(args.dataset_csv,fold_csv, 'train')
            SlideNames_val, _, _ = reOrganize_mDATA(args.dataset_csv, fold_csv, 'val')
            SlideNames_test, _, _ = reOrganize_mDATA(args.dataset_csv, fold_csv, 'val')

            # SlideNames_test, _, _ = read_test_data("/home/karan.padariya/CLAM/dataset_csv/updated_tcga-LUAD&LUSC_updated_modified.csv")

            train_path = ["temp_train/" +os.path.splitext(slide_id)[0]+".pt" for slide_id in SlideNames_train]
            val_path = ["temp_train/" +os.path.splitext(slide_id)[0]+".pt"  for slide_id in SlideNames_val]
            test_path = ["temp_train/" +os.path.splitext(slide_id)[0]+".pt"  for slide_id in SlideNames_test]

            fold_best_score = 0
            best_ac = 0
            best_auc = 0
            counter = 0
            if args.train:
                for epoch in range(1, args.num_epochs + 1):
                    counter += 1
                    train_loss_bag, train_acc, train_auc = train(args, train_path, milnet, criterion, optimizer, class_weights=class_weights)
                    val_loss_bag, avg_score, aucs, thresholds_optimal, val_auc,val_mat = test(args, val_path, milnet, criterion, class_weights=class_weights)
                    # write_test_metrics(val_mat,iteration)

                    print_epoch_info(epoch, args, train_loss_bag, train_acc, train_auc, val_loss_bag, avg_score, aucs, val_auc)
                    epoch_data = (epoch, args, train_loss_bag, train_acc, train_auc, val_loss_bag, avg_score, aucs, val_auc)
                    scheduler.step()

                    current_score = get_current_score(avg_score, aucs)
                    if current_score > fold_best_score:
                        counter = 0
                        fold_best_score = current_score
                        best_ac = avg_score
                        best_auc = aucs
                        best_val_mat = val_mat
                        save_model(args, iteration, run, save_path, milnet, thresholds_optimal, epoch_data)
                        best_model = copy.deepcopy(milnet)
                    if counter > args.stop_epochs: break
            else:
                milnet.load_state_dict(torch.load(f"{args.test_model_path}/fold_{iteration}_1.pth"))
                best_model  = copy.deepcopy(milnet)
            test_loss_bag, test_acc, test_aucs, _, test_auc, test_mat = test(args, test_path, best_model, criterion, class_weights=class_weights)
            file_name = os.path.join(save_path, f'fold_{iteration}_{run+1}_test.json')
            write_test_metrics(test_mat,iteration)

            data_to_save = {'test_loss': test_loss_bag,
                            'test_acc': test_acc,
                            'test_auc': test_auc,
                            **{f'test_auc_{i+1}': test_aucs[i] for i in range(args.num_classes)}}
            
            with open(file_name, 'a') as f:
                json.dump(data_to_save, f, indent=4)

            fold_results.append((best_ac, best_auc))
            
        mean_ac = np.mean(np.array([i[0] for i in fold_results]))
        mean_auc = np.mean(np.array([i[1] for i in fold_results]), axis=0)
        # Print mean and std deviation for each class
        print(f"Final results: Mean Accuracy: {mean_ac}")
        if args.train:
            for i, mean_score in enumerate(mean_auc):
                print(f"Class {i}: Mean AUC = {mean_score:.4f}")

    if args.eval_scheme == '10-fold-cv-standalone-test':
        bags_path = glob.glob('temp_train/*.pt')
        bags_path = shuffle(bags_path)
        reserved_testing_bags = bags_path[:int(args.split*len(bags_path))]
        bags_path = bags_path[int(args.split*len(bags_path)):]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []
        fold_models = []

        save_path = os.path.join('weights', datetime.date.today().strftime("%Y%m%d"))
        os.makedirs(save_path, exist_ok=True)
        run = len(glob.glob(os.path.join(save_path, '*.pth')))

        for fold, (train_index, test_index) in enumerate(kf.split(bags_path)):
            print(f"Starting CV fold {fold}.")
            milnet, criterion, optimizer, scheduler = init_model(args)
            train_path = [bags_path[i] for i in train_index]
            test_path = [bags_path[i] for i in test_index]
            fold_best_score = 0
            best_ac = 0
            best_auc = 0
            counter = 0
            best_model = []

            for epoch in range(1, args.num_epochs+1):
                counter += 1
                train_loss_bag = train(args, train_path, milnet, criterion, optimizer) # iterate all bags
                test_loss_bag, avg_score, aucs, thresholds_optimal = test(args, test_path, milnet, criterion)
                
                print_epoch_info(epoch, args, train_loss_bag, test_loss_bag, avg_score, aucs)
                scheduler.step()

                current_score = get_current_score(avg_score, aucs)
                if current_score > fold_best_score:
                    counter = 0
                    fold_best_score = current_score
                    best_ac = avg_score
                    best_auc = aucs
                    save_model(args, fold, run, save_path, milnet, thresholds_optimal)
                    best_model = [copy.deepcopy(milnet.cpu()), thresholds_optimal]
                    milnet.cuda()
                if counter > args.stop_epochs: break
            fold_results.append((best_ac, best_auc))
            fold_models.append(best_model)

        fold_predictions = []
        for item in fold_models:
            best_model = item[0]
            optimal_thresh = item[1]
            test_loss_bag, avg_score, aucs, thresholds_optimal, test_predictions, test_labels = test(args, reserved_testing_bags, best_model.cuda(), criterion, thresholds=optimal_thresh, return_predictions=True)
            fold_predictions.append(test_predictions)
        predictions_stack = np.stack(fold_predictions, axis=0)
        mode_result = mode(predictions_stack, axis=0)
        combined_predictions = mode_result.mode[0]
        combined_predictions = combined_predictions.squeeze()

        if args.num_classes > 1:
            # Compute Hamming Loss
            hammingloss = hamming_loss(test_labels, combined_predictions)
            print("Hamming Loss:", hammingloss)
            # Compute Subset Accuracy
            subset_accuracy = accuracy_score(test_labels, combined_predictions)
            print("Subset Accuracy (Exact Match Ratio):", subset_accuracy)
        else:
            accuracy = accuracy_score(test_labels, combined_predictions)
            print("Accuracy:", accuracy)
            balanced_accuracy = balanced_accuracy_score(test_labels, combined_predictions)
            print("Balanced Accuracy:", balanced_accuracy)

        os.makedirs('test', exist_ok=True)
        with open("test/test_list.json", "w") as file:
            json.dump(reserved_testing_bags, file)

        for i, item in enumerate(fold_models):
            best_model = item[0]
            optimal_thresh = item[1]
            torch.save(best_model.state_dict(), f"test/mil_weights_fold_{i}.pth")
            with open(f"test/mil_threshold_fold_{i}.json", "w") as file:
                optimal_thresh = [float(i) for i in optimal_thresh]
                json.dump(optimal_thresh, file)
                

if __name__ == '__main__':
    main()
