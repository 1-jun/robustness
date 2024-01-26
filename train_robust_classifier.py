#%%
import os
# import dill
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
import torchvision.transforms as T

import robustness

import robustness.datasets
import robustness.model_utils
import robustness.defaults
import robustness.train
from robustness import imagenet_models
from robustness.attacker import AttackerModel

import cox.utils
import cox.store

seed = 2228
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#%%
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=1.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class MIMIC_Dataset(torch.utils.data.Dataset):
    def __init__(self, mimic_path, df, transforms):
        """
        mimic_path (str):
            path to MIMIC CXRs (with subfolders p10, p11, ... p19)
        df (pd.DataFrame):
            DataFrame with columns 'dicom_id', 'subject_id',
            'study_id', and 'label'.
        """
        self.mimic_path = mimic_path
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        
        subject_id = row['subject_id']
        study_id = row['study_id']
        dicom_id = row['dicom_id']
        
        img_path = f"{self.mimic_path}/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        
        label = row['label']
        
        return img, label
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_labels', nargs='+',
                        default=['Pneumonia', 'No Finding',],
                        choices=['Atelectasis',
                                 'Cardiomegaly',
                                 'Consolidation',
                                 'Edema',
                                 'Enlarged Cardiomediastinum',
                                 'Fracture',
                                 'Lung Lesion',
                                 'Lung Opacity',
                                 'No Finding',
                                 'Pleural Effusion',
                                 'Pleural Other',
                                 'Pneumonia',
                                 'Pneumothorax',
                                 'Support Devices'],
                        help='target classes for classification')
    parser.add_argument('--balance_labels', type=bool, default=True,
                        help='Whether to balance the labels by selecting at random from the label with more data points')
    parser.add_argument('--save_path', type=str,
                        default='classifiers/test')
    parser.add_argument('--mimic_path', type=str,
                        default='/media/wonjun/HDD8TB/mimic-cxr-jpg-resized512')
    parser.add_argument('--batch_size', type=str, default=30)
    parser.add_argument('--arch', default='resnet50',
                        choices=['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
                                'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                                'wide_resnet50_2', 'wide_resnet101_2',
                                'wide_resnet50_3', 'wide_resnet50_4', 'wide_resnet50_5', 
                                'wide_resnet50_6', 'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
                                'vgg19_bn', 'vgg19', 'SqueezeNet', 'squeezenet1_0', 'squeezenet1_1',
                                'DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161',
                                'leaky_resnet18', 'leaky_resnet34', 'leaky_resnet50',
                                'leaky_resnet101', 'leaky_resnet152'],
                        help="Name of available architectures in robustness.imagenet_models")
    parser.add_argument('--adv_train', type=int, default=1,
                        help='1 for YES adversarial training; 0 for NO adversarial training')
    parser.add_argument('--constraint', type=str, default='2', choices=['1', '2', 'inf'],
                        help='2 for L2 constraint; I think there are also L1 constraint and L-inf constraint; not sure')
    parser.add_argument('--eps', type=int, default=3,
                        help='epsilon.. I think this is level of noise added for adversarial training? not sure')
    parser.add_argument('--attack_lr', type=float, default=0.1)
    parser.add_argument('--attack_steps', type=int, default=20,
                        help='adversarial attack steps. 20 is usually enough.')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save_ckpt_iters', type=int, default=10)
    
    args = parser.parse_args()
    return args

def add_metadata(df, metadata_path):
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df[metadata_df['ViewPosition'].isin(['PA', 'AP'])]
    
    study_ids = list(df['study_id'])
    study_ids_metadata = metadata_df[metadata_df['study_id'].isin(study_ids)]
    study_ids_metadata = study_ids_metadata[['dicom_id', 'subject_id', 'study_id']]
    
    df = pd.merge(study_ids_metadata, df, how='left',
                  on=['subject_id', 'study_id'])
    df = df.reset_index(drop=True)
    
    return df

# def create_2by2_for_mimic(metadata_path, labels_path,
#                             target="Edema", 
#                             confounder="Cardiac Devices"):

#     raw_df = pd.read_csv(labels_path)
#     raw_df = raw_df.fillna(0)
    
#     both_present = raw_df[(raw_df[target]==1) &
#                     (raw_df[confounder]==1)]
#     both_present = add_metadata(both_present, metadata_path)
    
#     target_only = raw_df[(raw_df[target]==1) &
#                     (raw_df[confounder]==0)]
#     target_only = add_metadata(target_only, metadata_path)
    
#     confounder_only = raw_df[(raw_df[target]==0) &
#                     (raw_df[confounder]==1)]
#     confounder_only = add_metadata(confounder_only, metadata_path)
    
#     both_absent = raw_df[(raw_df[target]==0) &
#                     (raw_df[confounder]==0)]
#     both_absent = add_metadata(both_absent, metadata_path)
    
#     return both_present, target_only, confounder_only, both_absent

def combine_into_labelled_df(
    df_both_present, df_target_only, df_confounder_only, df_both_absent,
    n_both_present, n_target_only,
    n_confounder_only, n_both_absent
):
    positive_label_df = pd.concat([
        df_both_present.sample(n=n_both_present),
        df_target_only.sample(n=n_target_only)
    ])
    negative_label_df = pd.concat([
        df_confounder_only.sample(n=n_confounder_only),
        df_both_absent.sample(n=n_both_absent)
    ])
    
    positive_label_df['label'] = 1
    negative_label_df['label'] = 0
    
    return positive_label_df, negative_label_df

# def print_2by2_table(df, target='Edema', confounder='Cardiac Devices'):
#     n_both_present = len(df[(df[target]==1) & (df[confounder]==1)])
#     n_both_absent = len(df[(df[target]==0) & (df[confounder]==0)])
#     n_target_only = len(df[(df[target]==1) & (df[confounder]==0)])
#     n_confounder_only = len(df[(df[target]==0) & (df[confounder]==1)])
    
#     table = [["   ", f"{target}(+)", f"{target}(-)"],
#              [f"{confounder}(+)", n_both_present, n_confounder_only],
#              [f"{confounder}(-)", n_target_only, n_both_absent]]
    
#     print(tabulate(table))
    

def train_valid_split(label_dataframes: list, n_folds):
        
    # df = pd.concat([positive_label_df, negative_label_df])
    df = pd.concat(label_dataframes)
    df = df.reset_index(drop=True)

    # Split into folds
    kfold = StratifiedKFold(
        n_splits = n_folds, shuffle=True, random_state=42
    )
    for i, (train_idx, valid_idx) in enumerate(kfold.split(df, y=df['label'])):
        df.loc[valid_idx, 'fold'] = i
    
    # pick one fold to be validation set and the rest training set
    validation_fold = 0
    train_df = df[df['fold'] != validation_fold]
    valid_df = df[df['fold'] == validation_fold]
    
    # print("2x2 table for train_df")
    # print_2by2_table(train_df)
    # print()
    # print("2x2 table for valid_df")
    # print_2by2_table(valid_df)
    
    return train_df, valid_df

def make_dataloaders(train_df, valid_df, mimic_path, batch_size):
    train_transforms = T.Compose(
        [
            T.RandomResizedCrop(256, scale=(0.97, 1.0), interpolation=Image.BICUBIC),
            # T.RandomRotation(degrees=(-5, 5)),
            # T.RandomAutocontrast(p=0.3),
            # T.RandomEqualize(p=0.3),
            # GaussianBlur(),
            T.ToTensor(),
        ]
    )

    valid_transforms = T.Compose(
        [
            T.Resize((256,256)),
            T.ToTensor(),
        ]
    )

    train_ds = MIMIC_Dataset(mimic_path, train_df, train_transforms)
    valid_ds = MIMIC_Dataset(mimic_path, valid_df, valid_transforms)
    print(f"Training set: {len(train_ds)}")
    print(f"Validation set: {len(valid_ds)}")

    train_sampler = RandomSampler(train_ds)
    valid_sampler = RandomSampler(valid_ds)

    train_loader = DataLoader(
        dataset = train_ds,
        sampler=train_sampler,
        batch_size = batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=False
    )
    valid_loader = DataLoader(
        dataset = valid_ds,
        sampler=valid_sampler,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, valid_loader


#%%
def train(args, train_loader, valid_loader):

    classifier_model = imagenet_models.__dict__[args.arch](num_classes=len(args.target_labels))

    # I know this looks stupid, but this is how the AttackerModel
    # class receives mean and SD for a dataset (ie, as attributes of a class)
    class MEAN_STD:
        def __init__(
            self,
            mean=torch.Tensor([0.5, 0.5, 0.5]),
            std=torch.Tensor([0.2, 0.2, 0.2])
        ):
            
            self.mean = mean
            self.std = std
    meanstd = MEAN_STD()
    model = AttackerModel(classifier_model, meanstd)

    # resume from a checkpoint
    # resume_path = '/home/wonjun/code/pnp/classifiers/robust_mimic_multilabel/checkpoint.pt.latest'
    # checkpoint = torch.load(resume_path, pickle_module=dill)
    # sd = checkpoint['model']
    # sd = {k[len('module.'):] : v for k, v in sd.items()}
    # model.load_state_dict(sd)
    # print(f"Loaded checkpoint from \n {resume_path}")

    train_kwargs = {
        'out_dir': args.save_path,
        'adv_train': args.adv_train, # 1 for YES adversarial training; 0 for NO
        'constraint': args.constraint, #2 for L2
        'eps': args.eps,
        'attack_lr': args.attack_lr,
        'attack_steps': args.attack_steps,
        'lr': args.lr,
        'epochs': args.epochs,
        'save_ckpt_iters':args.save_ckpt_iters
    }
    os.makedirs(train_kwargs['out_dir'], exist_ok=True)
    train_args = cox.utils.Parameters(train_kwargs)

    train_args = robustness.defaults.check_and_fill_args(train_args,
                                                        robustness.defaults.TRAINING_ARGS,
                                                        robustness.datasets.ImageNet)
    train_args = robustness.defaults.check_and_fill_args(train_args,
                                                        robustness.defaults.PGD_ARGS,
                                                        robustness.datasets.ImageNet)

    robustness.train.train_model(train_args, model,
                                (train_loader, valid_loader))
    
#%%
if __name__ == '__main__':
    args = parse_args()

    metadata_path = os.path.join(args.mimic_path, 'mimic-cxr-2.0.0-metadata.csv')
    labels_path = os.path.join(args.mimic_path, 'mimic-cxr-2.0.0-negbio.csv')
    labels = pd.read_csv(labels_path)
    labels = add_metadata(labels, metadata_path)

    # # If positive label for consolidation, positively label for pneumonia as well
    # labels.loc[labels['Consolidation']==1, 'Pneumonia'] = 1

    # We want 'No Finding' to have the label '0'
    # So we put it at the 0th index of the target_labels list
    if 'No Finding' in args.target_labels:
        ind = args.target_labels.index('No Finding')
        args.target_labels.insert(0, args.target_labels.pop(ind))

    label_dfs = []
    for i, label in enumerate(args.target_labels):
        df = labels[labels[label]==1.]
        df['label'] = i
        label_dfs.append(df)
    
    _label_dfs = []
    if args.balance_labels:
        num_rows = len(min(label_dfs, key=len))
        for df in label_dfs:
            _label_dfs.append(df.sample(num_rows))
    
    train_df, valid_df = train_valid_split(
        _label_dfs, n_folds=10
    )

    train_df.to_csv(os.path.join(args.save_path, 'train_df.csv'))
    valid_df.to_csv(os.path.join(args.save_path, 'valid_df.csv'))
    train_loader, valid_loader = make_dataloaders(
            train_df, valid_df,
            mimic_path=f"{args.mimic_path}/files",
            batch_size=args.batch_size
        )

    train(args, train_loader, valid_loader)