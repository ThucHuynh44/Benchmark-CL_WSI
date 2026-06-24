"""
mergeslide/datasets.py
Dataset classes for continual / lifelong learning on WSI feature bags.

Two dataset formats are supported:
  - Generic_MIL_Dataset  : BRCA, NSCLC, RCC — CSV-based with h5/pt feature files.
  - Generic_MIL_Dataset2 : ESCA, TGCT, CESC — split-CSV-based with h5 feature files.
"""

import bisect
import collections
import math
import os
import time
from abc import abstractmethod
from itertools import islice
from typing import List, Tuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as torchvision_datasets
from torchvision.transforms import transforms

from configs.loader import load_config


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ContinualDataset:
    """Abstract base class for continual learning dataset streams."""

    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self) -> None:
        self.train_loader = None
        self.test_loaders: List[DataLoader] = []
        self.i = 0

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create and return train/test loaders for the current task."""
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        pass

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        pass

    @staticmethod
    def get_epochs():
        pass

    @staticmethod
    def get_batch_size():
        pass


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def store_masked_loaders(
    train_dataset: torchvision_datasets,
    test_dataset: torchvision_datasets,
    setting: ContinualDataset,
) -> Tuple[DataLoader, DataLoader]:
    """Slice a dataset into a single task's train/test loaders."""
    train_mask = np.logical_and(
        np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK,
    )
    test_mask = np.logical_and(
        np.array(test_dataset.targets) >= setting.i,
        np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK,
    )
    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(train_dataset, batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader
    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


def get_previous_train_loader(
    train_dataset: torchvision_datasets, batch_size: int, setting: ContinualDataset
) -> DataLoader:
    """Create a DataLoader for the previous task's training data."""
    lo = setting.i - setting.N_CLASSES_PER_TASK
    hi = lo + setting.N_CLASSES_PER_TASK
    mask = np.logical_and(
        np.array(train_dataset.targets) >= lo,
        np.array(train_dataset.targets) < hi,
    )
    train_dataset.data = train_dataset.data[mask]
    train_dataset.targets = np.array(train_dataset.targets)[mask]
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def collate_MIL(batch):
    """Collate function for MIL bags: stack features, coords, and labels."""
    img = torch.cat([item[0] for item in batch], dim=0)
    coord = torch.cat([item[1] for item in batch], dim=0)
    label = torch.LongTensor([item[2] for item in batch])
    return [img, coord, label]


DEFAULT_TASK_ORDER = [
    "camelyon17",
    "brca",
    "rcc",
    "nsclc",
    "esca",
    "tgct",
    "cesc",
    "bracs",
    "herohe",
    "ubc_ocean",
]

TASK_SPECS = {
    "camelyon17": {
        "num_classes": 4,
        "label_dict": {"negative": 0, "itc": 1, "micro": 2, "macro": 3},
        "dataset_type": "annotation_csv",
        "label_col": "label",
    },
    "brca": {
        "num_classes": 2,
        "label_dict": {"IDC": 0, "ILC": 1},
        "dataset_type": "annotation_csv",
        "ignore": ["MDLC", "PD", "ACBC", "IMMC", "BRCNOS", "BRCA", "SPC", "MBC", "MPT"],
    },
    "rcc": {
        "num_classes": 3,
        "label_dict": {"CCRCC": 0, "PRCC": 1, "CHRCC": 2},
        "dataset_type": "annotation_csv",
    },
    "nsclc": {
        "num_classes": 2,
        "label_dict": {"LUAD": 0, "LUSC": 1},
        "dataset_type": "annotation_csv",
    },
    "esca": {
        "num_classes": 2,
        "label_dict": {0: 0, 1: 1},
        "dataset_type": "split_csv",
    },
    "tgct": {
        "num_classes": 2,
        "label_dict": {0: 0, 1: 1},
        "dataset_type": "split_csv",
    },
    "cesc": {
        "num_classes": 2,
        "label_dict": {0: 0, 1: 1},
        "dataset_type": "split_csv",
    },
    "bracs": {
        "num_classes": 3,
        "label_dict": {"Group_BT": 0, "Group_AT": 1, "Group_MT": 2},
        "dataset_type": "annotation_csv",
        "label_col": "label",
    },
    "herohe": {
        "num_classes": 2,
        "label_dict": {"Negative": 0, "Positive": 1},
        "dataset_type": "annotation_csv",
        "label_col": "label",
    },
    "ubc_ocean": {
        "num_classes": 5,
        "label_dict": {"HGSC": 0, "EC": 1, "CC": 2, "LGSC": 3, "MC": 4},
        "dataset_type": "annotation_csv",
        "label_col": "label",
    },
}


def get_task_specs(task_order=None):
    order = task_order or DEFAULT_TASK_ORDER
    return [TASK_SPECS[name] for name in order]


def get_num_classes(task_order=None):
    return [spec["num_classes"] for spec in get_task_specs(task_order)]


def get_dict_classes(num_classes):
    ranges, start = {}, 0
    for task_id, n_classes in enumerate(num_classes):
        ranges[task_id] = [start, start + n_classes - 1]
        start += n_classes
    return ranges


def get_dict_convert_class(num_classes):
    mapping, start = {}, 0
    for task_id, n_classes in enumerate(num_classes):
        mapping[task_id] = {local_id: start + local_id for local_id in range(n_classes)}
        start += n_classes
    return mapping


def _normalize_slide_id(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        prefix = text[:-2]
        if prefix.isdigit():
            text = prefix
    if text.endswith(".svs"):
        text = text[:-4]
    return text


def _slide_stem(value):
    return _normalize_slide_id(value)


def _debug_io_enabled():
    return os.environ.get("MERGESLIDE_DEBUG_IO", "").lower() in {"1", "true", "yes", "on"}


def _debug_io(message: str):
    if _debug_io_enabled():
        print(message, flush=True)


# ---------------------------------------------------------------------------
# Split generation helpers
# ---------------------------------------------------------------------------

def generate_split(cls_ids, val_num, test_num, samples, n_splits=5, seed=7, label_frac=1.0, custom_test_ids=None):
    indices = np.arange(samples).astype(int)
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)
    np.random.seed(seed)
    for _ in range(n_splits):
        all_val_ids, all_test_ids, sampled_train_ids = [], [], []
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
                sampled_train_ids.extend(remaining_ids[:sample_num])
        yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
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


# ---------------------------------------------------------------------------
# WSI Classification Dataset (CSV-based: BRCA, NSCLC, RCC)
# ---------------------------------------------------------------------------

class Generic_WSI_Classification_Dataset(Dataset):
    """Base WSI classification dataset loaded from a CSV annotation file."""

    def __init__(
        self,
        csv_path: str,
        shuffle: bool = False,
        seed: int = 7,
        print_info: bool = True,
        label_dict: dict = {},
        filter_dict: dict = {},
        ignore: list = [],
        patient_strat: bool = False,
        label_col: str = None,
        patient_voting: str = 'max',
    ):
        self.label_dict = label_dict
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = None, None, None
        self.data_dir = None
        self.label_col = label_col or 'oncotree_code'

        slide_data = pd.read_csv(csv_path)
        slide_data = self.filter_df(slide_data, filter_dict)
        slide_data = self.df_prep(slide_data, self.label_dict, ignore, self.label_col)
        slide_data['_split_id'] = slide_data['slide_id'].map(_normalize_slide_id)
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)
        self.slide_data = slide_data
        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

    def cls_ids_prep(self):
        self.patient_cls_ids = [
            np.where(self.patient_data['label'] == i)[0] for i in range(self.num_classes)
        ]
        self.slide_cls_ids = [
            np.where(self.slide_data['label'] == i)[0] for i in range(self.num_classes)
        ]

    def patient_data_prep(self, patient_voting: str = 'max'):
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
        data = data[~data['label'].isin(ignore)].reset_index(drop=True)
        data['label'] = data['label'].map(label_dict)
        
        # Kiểm tra lỗi (safety check) tránh chuyển nhãn lỗi thành NaN im lặng
        nan_mask = data['label'].isna()
        if nan_mask.any():
            invalid_labels = data.loc[nan_mask, label_col].unique().tolist()
            raise ValueError(
                f"Lỗi: Tìm thấy các nhãn không hợp lệ hoặc chưa được định nghĩa trong label_dict: {invalid_labels}. "
                f"Vui lòng kiểm tra lại file CSV annotation hoặc cấu hình label_dict."
            )
        return data

    def filter_df(self, df, filter_dict: dict = {}):
        if filter_dict:
            filter_mask = np.full(len(df), True, bool)
            for key, val in filter_dict.items():
                filter_mask = np.logical_and(filter_mask, df[key].isin(val))
            df = df[filter_mask]
        return df

    def __len__(self):
        return len(self.patient_data['case_id']) if self.patient_strat else len(self.slide_data)

    def summarize(self):
        print(f"label column: {self.label_col}")
        print(f"label dictionary: {self.label_dict}")
        print(f"number of classes: {self.num_classes}")
        print("slide-level counts:\n", self.slide_data['label'].value_counts(sort=False))
        for i in range(self.num_classes):
            print(f"Patient-LVL; Number of samples registered in class {i}: {self.patient_cls_ids[i].shape[0]}")
            print(f"Slide-LVL; Number of samples registered in class {i}: {self.slide_cls_ids[i].shape[0]}")

    def create_splits(self, k=3, val_num=(25, 25), test_num=(40, 40), label_frac=1.0, custom_test_ids=None):
        settings = {
            'n_splits': k, 'val_num': val_num, 'test_num': test_num,
            'label_frac': label_frac, 'seed': self.seed, 'custom_test_ids': custom_test_ids,
        }
        if self.patient_strat:
            settings.update({'cls_ids': self.patient_cls_ids, 'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'cls_ids': self.slide_cls_ids, 'samples': len(self.slide_data)})
        self.split_gen = generate_split(**settings)

    def set_splits(self, start_from=None):
        ids = nth(self.split_gen, start_from) if start_from else next(self.split_gen)
        if self.patient_strat:
            slide_ids = [[] for _ in range(len(ids))]
            for split in range(len(ids)):
                for idx in ids[split]:
                    case_id = self.patient_data['case_id'][idx]
                    slide_indices = self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    slide_ids[split].extend(slide_indices)
            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]
        else:
            self.train_ids, self.val_ids, self.test_ids = ids

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key].dropna().reset_index(drop=True)
        if len(split) > 0:
            split_ids = {_normalize_slide_id(i) for i in split.tolist()}
            slide_ids = self.slide_data.get('_split_id', self.slide_data['slide_id'].map(_normalize_slide_id))
            mask = slide_ids.isin(split_ids)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            return Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
        return None

    def get_merged_split_from_df(self, all_splits, split_keys=('train',)):
        merged_split = []
        for key in split_keys:
            merged_split.extend(all_splits[key].dropna().reset_index(drop=True).tolist())
        if merged_split:
            split_ids = {_normalize_slide_id(i) for i in merged_split}
            slide_ids = self.slide_data.get('_split_id', self.slide_data['slide_id'].map(_normalize_slide_id))
            mask = slide_ids.isin(split_ids)
            df_slice = self.slide_data[mask].reset_index(drop=True)
            return Generic_Split(df_slice, data_dir=self.data_dir, num_classes=self.num_classes)
        return None

    def return_splits(self, from_id=True, csv_path=None):
        if from_id:
            def _make_split(ids):
                if len(ids) > 0:
                    data = self.slide_data.loc[ids].reset_index(drop=True)
                    return Generic_Split(data, data_dir=self.data_dir, num_classes=self.num_classes)
                return None
            return _make_split(self.train_ids), _make_split(self.val_ids), _make_split(self.test_ids)
        else:
            assert csv_path
            all_splits = pd.read_csv(csv_path, dtype=str)
            return (
                self.get_split_from_df(all_splits, 'train'),
                self.get_split_from_df(all_splits, 'val'),
                self.get_split_from_df(all_splits, 'test'),
            )

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
        for split_key, ids in [('train', self.train_ids), ('val', self.val_ids), ('test', self.test_ids)]:
            labels = self.getlabel(ids)
            unique, counts = np.unique(labels, return_counts=True)
            if return_descriptor:
                for u, c in zip(unique, counts):
                    df.loc[index[u], split_key] = c
        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0
        if return_descriptor:
            return df

    def save_split(self, filename):
        df = pd.concat([
            pd.DataFrame({'train': self.get_list(self.train_ids)}),
            pd.DataFrame({'val': self.get_list(self.val_ids)}),
            pd.DataFrame({'test': self.get_list(self.test_ids)}),
        ], axis=1)
        df.to_csv(filename, index=False)


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    """MIL dataset for BRCA/NSCLC/RCC — loads features from h5 or pt files."""

    def __init__(self, data_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.use_h5 = True

    def load_from_h5(self, toggle: bool):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        label = self.slide_data['label'][idx]
        stem = _slide_stem(slide_id)
        debug = _debug_io_enabled()
        start = time.time() if debug else None
        h5_candidates = [
            os.path.join(self.data_dir, 'h5_files', f"{stem}.h5"),
            os.path.join(self.data_dir, 'features_conch_v15', f"{stem}.h5"),
            os.path.join(self.data_dir, f"{stem}.h5"),
        ]
        pt_candidates = [
            os.path.join(self.data_dir, 'pt_files', f"{stem}.pt"),
            os.path.join(self.data_dir, f"{stem}.pt"),
        ]

        features, coords = None, None
        coords_from_h5 = None
        h5_missing_features = []
        source_path = None
        if debug:
            _debug_io(f"[IO start] idx={idx} slide_id={slide_id} stem={stem} label={label}")
        for h5_path in h5_candidates:
            if os.path.exists(h5_path):
                source_path = h5_path
                if debug:
                    _debug_io(f"[IO h5 open] {h5_path}")
                with h5py.File(h5_path, 'r') as f:
                    h5_coords = f['coords'][:] if 'coords' in f else None
                    if h5_coords is not None:
                        coords_from_h5 = h5_coords
                    if 'features' not in f:
                        h5_missing_features.append((h5_path, list(f.keys())))
                        continue
                    features = f['features'][:]
                    coords = h5_coords if h5_coords is not None else coords_from_h5
                break

        if features is None:
            for pt_path in pt_candidates:
                if os.path.exists(pt_path):
                    source_path = pt_path
                    if debug:
                        _debug_io(f"[IO pt load] {pt_path}")
                    features = torch.load(pt_path, map_location='cpu')
                    if isinstance(features, dict):
                        coords = features.get('coords')
                        features = features.get('features', features.get('feat', features))
                    break

        if features is None:
            checked = h5_candidates + pt_candidates
            raise FileNotFoundError(
                f"No usable feature file found for slide_id={slide_id}. "
                f"Checked: {checked}. H5 files missing 'features': {h5_missing_features}"
            )

        if not torch.is_tensor(features):
            features = torch.from_numpy(features)
        if coords is None and coords_from_h5 is not None:
            coords = coords_from_h5
        if coords is not None and not torch.is_tensor(coords):
            coords = torch.from_numpy(coords)
        if coords is not None and coords.shape[0] != features.shape[0]:
            raise ValueError(
                f"Feature/coord length mismatch for slide_id={slide_id}: "
                f"features={features.shape[0]}, coords={coords.shape[0]}"
            )
        if coords is None:
            coords = torch.zeros((features.shape[0], 2), dtype=torch.long)
        if debug:
            _debug_io(
                f"[IO done] idx={idx} slide_id={slide_id} source={source_path} "
                f"features={tuple(features.shape)} coords={tuple(coords.shape)} "
                f"elapsed={time.time() - start:.3f}s"
            )
        return features, coords, label


class Generic_Split(Generic_MIL_Dataset):
    """A pre-split subset of Generic_MIL_Dataset."""

    def __init__(self, slide_data, data_dir=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.slide_cls_ids = [
            np.where(self.slide_data['label'] == i)[0] for i in range(self.num_classes)
        ]

    def __len__(self):
        return len(self.slide_data)


# ---------------------------------------------------------------------------
# Alternative MIL Dataset (split-CSV-based: ESCA, TGCT, CESC)
# ---------------------------------------------------------------------------

class Generic_MIL_Dataset2:
    """MIL dataset for ESCA/TGCT/CESC — reads train/val/test from a split CSV."""

    def __init__(self, data_dir: str, label_dict: dict):
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.use_h5 = True

    def return_splits(self, from_id=False, csv_path=None):
        slide_data = pd.read_csv(csv_path, index_col=0)
        data_train  = list(slide_data['train'].dropna())
        label_train = [self.label_dict[int(l)] for l in slide_data['train_label'].dropna()]
        data_val    = list(slide_data['val'].dropna())
        label_val   = [self.label_dict[int(l)] for l in slide_data['val_label'].dropna()]
        data_test   = list(slide_data['test'].dropna())
        label_test  = [self.label_dict[int(l)] for l in slide_data['test_label'].dropna()]
        return (
            Generic_MIL_Dataset2_Split(self.data_dir, data_train, label_train),
            Generic_MIL_Dataset2_Split(self.data_dir, data_val, label_val),
            Generic_MIL_Dataset2_Split(self.data_dir, data_test, label_test),
        )


class Generic_MIL_Dataset2_Split:
    """A split subset for Generic_MIL_Dataset2."""

    def __init__(self, data_dir: str, data: list, label: list):
        self.data_dir = data_dir
        self.data = data
        self.label = label

    def load_from_h5(self, toggle: bool):
        self.use_h5 = toggle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        slide_id = self.data[idx]
        label = self.label[idx]
        h5_path = os.path.join(self.data_dir, 'h5_files', f"{slide_id}.h5")
        debug = _debug_io_enabled()
        start = time.time() if debug else None
        if debug:
            _debug_io(f"[IO start] idx={idx} slide_id={slide_id} label={label}")
            _debug_io(f"[IO h5 open] {h5_path}")
        with h5py.File(h5_path, 'r') as f:
            features = torch.from_numpy(f['features'][:])
            coords = torch.from_numpy(f['coords'][:])
        if debug:
            _debug_io(
                f"[IO done] idx={idx} slide_id={slide_id} source={h5_path} "
                f"features={tuple(features.shape)} coords={tuple(coords.shape)} "
                f"elapsed={time.time() - start:.3f}s"
            )
        return features, coords, label


# ---------------------------------------------------------------------------
# Concat helper
# ---------------------------------------------------------------------------

class ConcatDataset(Dataset):
    """Concatenate multiple datasets on-the-fly."""

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            s += len(e)
            r.append(s)
        return r

    def __init__(self, datasets):
        super().__init__()
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
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


# ---------------------------------------------------------------------------
# Sequential stream of configured WSI tasks
# ---------------------------------------------------------------------------

class Sequential_Generic_MIL_Dataset(ContinualDataset):
    """Continual learning stream configured by task_order in datasets.yaml.

    Dataset paths are read from configs/datasets.yaml (git-ignored).
    Copy configs/datasets.yaml.example → configs/datasets.yaml and set your data_root.
    You can also point to a custom config via the MERGESLIDE_CONFIG env variable.
    """

    NAME = 'seq-wsi'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = len(DEFAULT_TASK_ORDER)
    TRANSFORM = None

    def __init__(self, config_path: str = None):
        """Args:
            config_path: Optional path to a YAML config file.
                         Defaults to configs/datasets.yaml (or MERGESLIDE_CONFIG env var).
        """
        super().__init__()
        cfg = load_config(config_path, default_filename="train.yaml")
        self.task_order = cfg.get("task_order", DEFAULT_TASK_ORDER)
        self.task_specs = get_task_specs(self.task_order)
        self.num_classes = [spec["num_classes"] for spec in self.task_specs]
        self.N_TASKS = len(self.task_order)
        self.datasets = []
        for task_name, spec in zip(self.task_order, self.task_specs):
            data_dir = cfg["features"].get(task_name, "")
            if spec["dataset_type"] == "split_csv":
                self.datasets.append(
                    Generic_MIL_Dataset2(data_dir=data_dir, label_dict=spec["label_dict"])
                )
            else:
                csv_path = cfg["annotations"].get(task_name, "")
                self.datasets.append(
                    Generic_MIL_Dataset(
                        csv_path=csv_path,
                        data_dir=data_dir,
                        shuffle=False,
                        seed=0,
                        print_info=True,
                        label_dict=spec["label_dict"],
                        patient_strat=False,
                        ignore=spec.get("ignore", []),
                        label_col=spec.get("label_col"),
                    )
                )
        # load_config() đã trả về split_dirs là list theo task_order (loader.py dòng 111)
        # Không cần index lại bằng tên — gán trực tiếp
        self.split_dirs = cfg['split_dirs']
        dataloader_cfg = cfg.get("dataloader", {})
        self.num_workers = int(
            os.environ.get("MERGESLIDE_NUM_WORKERS", dataloader_cfg.get("num_workers", 4))
        )

    def get_data_loaders(self, fold: int, task_id: int, num_workers: int = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Return (train, val, test) loaders for a given fold and task."""
        dataset = self.datasets[task_id]
        split_csv = f"{self.split_dirs[task_id]}/splits_{fold}.csv"
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path=split_csv)
        workers = self.num_workers if num_workers is None else int(num_workers)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=workers, collate_fn=collate_MIL)
        val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=True, num_workers=workers, collate_fn=collate_MIL)
        test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, num_workers=workers, collate_fn=collate_MIL)
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
        self.val_loader = val_loader
        return train_loader, val_loader, test_loader

    def get_joint_data_loaders(self, fold: int) -> Tuple[DataLoader, DataLoader, List[DataLoader]]:
        """Return joint loaders over all tasks (for joint/multi-task baseline).
        Returns (train_loader, val_loader, test_loaders) where test_loaders is a
        list of per-task test loaders (one per task).
        """
        train_datasets, val_datasets, all_test_loaders = [], [], []
        for n in range(self.N_TASKS):
            print(f"Loading dataset {n}")
            dataset = self.datasets[n]
            split_csv = f"{self.split_dirs[n]}/splits_{fold}.csv"
            train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path=split_csv)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=collate_MIL)
            all_test_loaders.append(test_loader)
            self.test_loaders.append(test_loader)

        train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=1, shuffle=True, num_workers=self.num_workers, collate_fn=collate_MIL)
        val_loader   = DataLoader(ConcatDataset(val_datasets),   batch_size=1, shuffle=True, num_workers=self.num_workers, collate_fn=collate_MIL)
        self.i = self.N_CLASSES_PER_TASK * self.N_TASKS
        self.train_loader = train_loader
        self.val_loader = val_loader
        return train_loader, val_loader, all_test_loaders


if __name__ == '__main__':
    seq_dataset = Sequential_Generic_MIL_Dataset()
    trains, vals, tests = seq_dataset.get_data_loaders(fold=0, task_id=0)
