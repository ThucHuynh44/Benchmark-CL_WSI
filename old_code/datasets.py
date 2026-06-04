from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import datasets
import numpy as np
import torch.optim
import os
import torch
import numpy as np
import pandas as pd
import math
from scipy import stats
import collections
from itertools import islice
import bisect
from torch.utils.data import Dataset
import h5py
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple

class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass

    @staticmethod
    def get_minibatch_size():
        pass

def store_masked_loaders(train_dataset: datasets, test_dataset: datasets, setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i, np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i, np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]
    train_loader = DataLoader(train_dataset, batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader
    setting.i += setting.N_CLASSES_PER_TASK
    return (train_loader, test_loader)

def get_previous_train_loader(train_dataset: datasets, batch_size: int, setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets) < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)
    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim=0)
    coord = torch.cat([item[1] for item in batch], dim=0)
    label = torch.LongTensor([item[2] for item in batch])
    return [img, coord, label]

def generate_split(cls_ids, val_num, test_num, samples, n_splits=5, seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)
    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []
        if custom_test_ids is not None:
            all_test_ids.extend(custom_test_ids)
        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices)
            val_ids = np.random.choice(possible_indices, val_num[c], replace=False)
            remaining_ids = np.setdiff1d(possible_indices, val_ids)
            all_val_ids.extend(val_ids)
            if custom_test_ids is None:
                test_ids = np.random.choice(remaining_ids, test_num[c], replace=False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)
            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)
            else:
                sample_num = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])
        yield (sampled_train_ids, all_val_ids, all_test_ids)

def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator, n, None), default)

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=['train', 'val', 'test'])
    df.to_csv(filename)
    print()

class Generic_WSI_Classification_Dataset(Dataset):

    def __init__(self, csv_path='dataset_csv/ccrcc_clean.csv', shuffle=False, seed=7, print_info=True, label_dict={}, filter_dict={}, ignore=[], patient_strat=False, label_col=None, patient_voting='max'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        if not label_col:
            label_col = 'oncotree_code'
        self.label_col = label_col
        slide_data = pd.read_csv(csv_path)
        slide_data = self.filter_df(slide_data, filter_dict)
        slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)
        self.slide_data = slide_data
        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def patient_data_prep(self, patient_voting='max'):
        patients = np.unique(np.array(self.slide_data['case_id']))
        patient_labels = []
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations].values
            if patient_voting == 'max':
                label = label.max()
            elif patient_voting == 'maj':
                label = stats.mode(label)[0]
            else:
                raise NotImplementedError
            patient_labels.append(label)
        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()
        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]
        return data

    def filter_df(self, df, filter_dict={}):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print('label column: {}'.format(self.label_col))
        print('label dictionary: {}'.format(self.label_dict))
        print('number of classes: {}'.format(self.num_classes))
        print('slide-level counts: ', '\n', self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    def create_splits(self, k=3, val_num=(25, 25), test_num=(40, 40), label_frac=1.0, custom_test_ids=None):
        settings = {'n_splits': k, 'val_num': val_num, 'test_num': test_num, 'label_frac': label_frac, 'seed': self.seed, 'custom_test_ids': custom_test_ids}
        if self.patient_strat:
            settings.update({'cls_ids': self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'cls_ids': self.slide_cls_ids, 'samples': len(self.slide_data)})
        self.split_gen = generate_split(**settings)

    def set_splits(self, start_from=None):
        if start_from:
            ids = nth(self.split_gen, start_from)
        else:
            ids = next(self.split_gen)
        if self.patient_strat:
            slide_ids = [[] for i in range(len(ids))]
            for split in range(len(ids)):
                for idx in ids[split]:
                    case_id = self.patient_data['case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    slide_ids[split].extend(slide_indices)
            self.train_ids, self.val_ids, self.test_ids = (slide_ids[0], slide_ids[1], slide_ids[2])
        else:
            self.train_ids, self.val_ids, self.test_ids = ids

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)
        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin([i + '.svs' for i in split.tolist()])
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
        else:
            split = None
        return split

    def get_merged_split_from_df(self, all_splits, split_keys=['train']):
        merged_split = []
        for split_key in split_keys:
            split = all_splits[split_key]
            split = split.dropna().reset_index(drop=True).tolist()
            merged_split.extend(split)
        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(merged_split)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
        else:
            split = None
        return split

    def return_splits(self, from_id=True, csv_path=None):
        if from_id:
            if len(self.train_ids) > 0:
                train_data = self.slide_data.loc[self.train_ids].reset_index(drop=True)
                train_split = Generic_Split(train_data, data_dir=self.data_dir, num_classes=self.num_classes)
            else:
                train_split = None
            if len(self.val_ids) > 0:
                val_data = self.slide_data.loc[self.val_ids].reset_index(drop=True)
                val_split = Generic_Split(val_data, data_dir=self.data_dir, num_classes=self.num_classes)
            else:
                val_split = None
            if len(self.test_ids) > 0:
                test_data = self.slide_data.loc[self.test_ids].reset_index(drop=True)
                test_split = Generic_Split(test_data, data_dir=self.data_dir, num_classes=self.num_classes)
            else:
                test_split = None
        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
            train_split = self.get_split_from_df(all_splits, 'train')
            val_split = self.get_split_from_df(all_splits, 'val')
            test_split = self.get_split_from_df(all_splits, 'test')
        return (train_split, val_split, test_split)

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def test_split_gen(self, return_descriptor=False):
        if return_descriptor:
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)] for i in range(self.num_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.full((len(index), len(columns)), 0, dtype=np.int32), index=index, columns=columns)
        count = len(self.train_ids)
        labels = self.getlabel(self.train_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            if return_descriptor:
                df.loc[index[u], 'train'] = counts[u]
        count = len(self.val_ids)
        labels = self.getlabel(self.val_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            if return_descriptor:
                df.loc[index[u], 'val'] = counts[u]
        count = len(self.test_ids)
        labels = self.getlabel(self.test_ids)
        unique, counts = np.unique(labels, return_counts=True)
        for u in range(len(unique)):
            if return_descriptor:
                df.loc[index[u], 'test'] = counts[u]
        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0
        if return_descriptor:
            return df

    def save_split(self, filename):
        train_split = self.get_list(self.train_ids)
        val_split = self.get_list(self.val_ids)
        test_split = self.get_list(self.test_ids)
        df_tr = pd.DataFrame({'train': train_split})
        df_v = pd.DataFrame({'val': val_split})
        df_t = pd.DataFrame({'test': test_split})
        df = pd.concat([df_tr, df_v, df_t], axis=1)
        df.to_csv(filename, index=False)

class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):

    def __init__(self, data_dir, **kwargs):
        super(Generic_MIL_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = True

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        data_dir = self.data_dir
        full_path = os.path.join(data_dir, 'h5_files', '{}.h5'.format(slide_id.split('.svs')[0]))
        with h5py.File(full_path, 'r') as hdf5_file:
            try:
                features = hdf5_file['features'][:]
                coords = hdf5_file['coords'][:]
            except:
                features = torch.load(os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.split('.svs')[0])))
                coords = hdf5_file['coords'][:]
        
        try:
            features = torch.from_numpy(features)
        except:
            pass
            
        coords = torch.from_numpy(coords)
        return (features, coords, label)

class Generic_MIL_Dataset2:

    def __init__(self, data_dir, label_dict):
        super(Generic_MIL_Dataset2, self).__init__()
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.use_h5 = True

    def return_splits(self, from_id=False, csv_path=None):
        slide_data = pd.read_csv(csv_path, index_col=0)
        self.data_train = [filename for filename in slide_data.loc[:, 'train'].dropna()]
        self.label_train = [self.label_dict[int(l)] for l in slide_data.loc[:, 'train_label'].dropna()]
        self.data_val = [filename for filename in slide_data.loc[:, 'val'].dropna()]
        self.label_val = [self.label_dict[int(l)] for l in slide_data.loc[:, 'val_label'].dropna()]
        self.data_test = [filename for filename in slide_data.loc[:, 'test'].dropna()]
        self.label_test = [self.label_dict[int(l)] for l in slide_data.loc[:, 'test_label'].dropna()]
        return (Generic_MIL_Dataset2_Split(self.data_dir, self.data_train, self.label_train), Generic_MIL_Dataset2_Split(self.data_dir, self.data_val, self.label_val), Generic_MIL_Dataset2_Split(self.data_dir, self.data_test, self.label_test))

class Generic_MIL_Dataset2_Split:

    def __init__(self, data_dir, data, label):
        super(Generic_MIL_Dataset2_Split, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.label = label

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = self.label[idx]
        data_dir = self.data_dir
        full_path = os.path.join(data_dir, 'h5_files', '{}.h5'.format(slide_id))
        with h5py.File(full_path, 'r') as hdf5_file:
            features = hdf5_file['features'][:]
            coords = hdf5_file['coords'][:]
        features = torch.from_numpy(features)
        coords = torch.from_numpy(coords)
        return (features, coords, label)

class Generic_Split(Generic_MIL_Dataset):

    def __init__(self, slide_data, data_dir=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    def __len__(self):
        return len(self.slide_data)

class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = ([], 0)
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError('absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

class Sequential_Generic_MIL_Dataset(ContinualDataset):
    NAME = 'seq-wsi'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 6
    TRANSFORM = None
    datasets = [Generic_MIL_Dataset(csv_path='/home/bui/datasets/wsi_dataset_annotation/tcga_brca/tcga_brca_subset.csv.zip', data_dir='/home/bui/datasets/TCGA-BRCA_processed/features/', shuffle=False, seed=0, print_info=True, label_dict={'IDC': 0, 'ILC': 1}, patient_strat=False, ignore=['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT']), 
                Generic_MIL_Dataset(csv_path='/home/bui/datasets/wsi_dataset_annotation/tcga_rcc/tcga_kidney_subset.csv.zip', data_dir='/home/bui/datasets/TCGA-RCC_processed/features/', shuffle=False, seed=0, print_info=True, label_dict={'CCRCC': 0, 'PRCC': 1, 'CHRCC': 2}, patient_strat=False, ignore=[]), 
                Generic_MIL_Dataset(csv_path='/home/bui/datasets/wsi_dataset_annotation/tcga_nsclc/tcga_lung_subset.csv.zip', data_dir='/home/bui/datasets/TCGA-NSCLC_processed/features/', shuffle=False, seed=0, print_info=True, label_dict={'LUAD': 0, 'LUSC': 1}, patient_strat=False, ignore=[]), 
                Generic_MIL_Dataset2(data_dir='/home/bui/datasets/TCGA-ESCA_processed/features/', label_dict={0: 0, 1: 1}), 
                Generic_MIL_Dataset2(data_dir='/home/bui/datasets/TCGA-TGCT_processed/features/', label_dict={0: 0, 1: 1}), 
                Generic_MIL_Dataset2(data_dir='/home/bui/datasets/TCGA-CESC_processed/features/', label_dict={0: 0, 1: 1})]
    
    split_dirs = ['/home/bui/datasets/wsi_dataset_annotation/tcga_brca', '/home/bui/datasets/wsi_dataset_annotation/tcga_rcc', '/home/bui/datasets/wsi_dataset_annotation/tcga_nsclc', '/home/bui/datasets/wsi_dataset_annotation/tcga_esca', '/home/bui/datasets/wsi_dataset_annotation/tcga_tgct', '/home/bui/datasets/wsi_dataset_annotation/tcga_cesc']

    def get_data_loaders(self, FOLD, task_id):
        dataset = self.datasets[task_id]
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(self.split_dirs[task_id], FOLD))
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_MIL)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_MIL)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_MIL)
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
        self.val_loader = val_loader
        return (train_loader, val_loader, test_loader)

    def get_joint_data_loaders(self, FOLD):
        train_datasets, val_datasets, test_datasets = ([], [], [])
        for n in range(self.N_TASKS):
            print('Loading dataset ', n)
            dataset = self.datasets[n]
            train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(self.split_dirs[n], FOLD))
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_MIL)
            self.test_loaders.append(test_loader)
        
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_MIL)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_MIL)
        
        self.i = self.N_CLASSES_PER_TASK * self.N_TASKS
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        return (train_loader, val_loader, test_loader)
    
if __name__ == '__main__':
    seq_dataset = Sequential_Generic_MIL_Dataset()
    fold = 0
    task_id = 0
    trains, vals, tests = seq_dataset.get_data_loaders(fold, task_id)