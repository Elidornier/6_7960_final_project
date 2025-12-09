# Standard Library Modules
import os
import sys
import pickle
import argparse
# 3rd-party Modules
import pandas as pd
from tqdm.auto import tqdm
# Pytorch Modules
import torch
import torchvision
# Huggingface Modules
from datasets import load_dataset
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.utils import check_path

def load_data(args: argparse.Namespace):

    name = args.task_dataset.lower()
    train_valid_split = args.train_valid_split

    def apply_data_fraction(df, fraction, seed):
        if fraction < 1.0:
            # Randomly sample the fraction using the seed for reproducibility
            return df.sample(frac=fraction, random_state=seed).reset_index(drop=True)
        return df

    train_data = {
        'image': [],
        'label': []
    }
    valid_data = {
        'image': [],
        'label': []
    }
    test_data = {
        'image': [],
        'label': []
    }

    if name == 'mnist':
        dataset = load_dataset('mnist')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 10

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['image'] = train_df['image'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['image'] = valid_df['image'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['image'] = test_df['image'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'fashion_mnist':
        dataset = load_dataset('fashion_mnist')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 10

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['image'] = train_df['image'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['image'] = valid_df['image'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['image'] = test_df['image'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'cifar10':
        dataset = load_dataset('cifar10')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 10

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        if hasattr(args, 'data_fraction') and args.data_fraction < 1.0:
            seed = args.seed if args.seed is not None else 9297
            print(f"Subsampling {name} Training Data to {args.data_fraction*100}%...")
            train_df = apply_data_fraction(train_df, args.data_fraction, seed)
            print(f"New Training Size: {len(train_df)}")

        train_data['image'] = train_df['img'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['image'] = valid_df['img'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['image'] = test_df['img'].tolist()
        test_data['label'] = test_df['label'].tolist()
    elif name == 'cifar100':
        dataset = load_dataset('cifar100')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No pre-defined validation set
        test_df = pd.DataFrame(dataset['test'])
        num_classes = 100

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        # What I added for data fraction
        # This ensures we only shrink the training set, not the validation set
        if hasattr(args, 'data_fraction') and args.data_fraction < 1.0:
            seed = args.seed if args.seed is not None else 9297
            print(f"Subsampling CIFAR-100 Training Data to {args.data_fraction*100}%...")
            train_df = apply_data_fraction(train_df, args.data_fraction, seed)
            print(f"New Training Size: {len(train_df)}")

        train_data['image'] = train_df['img'].tolist()
        train_data['label'] = train_df['fine_label'].tolist()
        valid_data['image'] = valid_df['img'].tolist()
        valid_data['label'] = valid_df['fine_label'].tolist()
        test_data['image'] = test_df['img'].tolist()
        test_data['label'] = test_df['fine_label'].tolist()
    elif name == 'tiny_imagenet':
        dataset = load_dataset('zh-plus/tiny-imagenet')

        train_df = pd.DataFrame(dataset['train'])
        #valid_df = pd.DataFrame(dataset['validation']) # No public test set -> Use validation set as test set
        test_df = pd.DataFrame(dataset['valid']) # No public test set
        num_classes = 200

        # train-valid split
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        valid_df = train_df[:int(len(train_df) * train_valid_split)]
        train_df = train_df[int(len(train_df) * train_valid_split):]

        train_data['image'] = train_df['image'].tolist()
        train_data['label'] = train_df['label'].tolist()
        valid_data['image'] = valid_df['image'].tolist()
        valid_data['label'] = valid_df['label'].tolist()
        test_data['image'] = test_df['image'].tolist()
        test_data['label'] = test_df['label'].tolist()

    return train_data, valid_data, test_data, num_classes

def preprocessing(args: argparse.Namespace) -> None:

    # Load data
    train_data, valid_data, test_data, num_classes = load_data(args)

    # Preprocessing - Define data_dict
    data_dict = {
        'train': {
            'images': train_data['image'],
            'labels': train_data['label'],
            'num_classes': num_classes,
        },
        'valid': {
            'images': valid_data['image'],
            'labels': valid_data['label'],
            'num_classes': num_classes,
        },
        'test': {
            'images': test_data['image'],
            'labels': test_data['label'],
            'num_classes': num_classes,
        }
    }

    # Save data as pickle file
    preprocessed_path = os.path.join(args.preprocess_path, args.task, args.task_dataset, args.model_type)
    check_path(preprocessed_path)

    for split_data, split in zip([train_data, valid_data, test_data], ['train', 'valid', 'test']):
        # Save data as pickle file
        with open(os.path.join(preprocessed_path, f'{split}_processed.pkl'), 'wb') as f:
            pickle.dump(data_dict[split], f)
