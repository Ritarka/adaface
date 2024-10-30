import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd
import evaluate_utils
from dataset.index_dataset import IndexedDatasetWrapper
from dataset.image_folder_dataset import CustomImageFolderDataset
from dataset.five_validation_dataset import FiveValidationDataset
from dataset.record_dataset import AugmentRecordDataset


class DataModule(pl.LightningDataModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.output_dir = kwargs['output_dir']
        self.data_root = kwargs['data_root']
        self.train_data_path = kwargs['train_data_path']
        self.val_data_path = kwargs['val_data_path']
        self.batch_size = kwargs['batch_size']
        self.num_workers = kwargs['num_workers']
        self.train_data_subset = kwargs['train_data_subset']

        self.low_res_augmentation_prob = kwargs['low_res_augmentation_prob']
        self.crop_augmentation_prob = kwargs['crop_augmentation_prob']
        self.photometric_augmentation_prob = kwargs['photometric_augmentation_prob']
        self.swap_color_channel = kwargs['swap_color_channel']
        self.use_mxrecord = kwargs['use_mxrecord']

        concat_mem_file_name = os.path.join(self.data_root, self.val_data_path, 'concat_validation_memfile')
        self.concat_mem_file_name = concat_mem_file_name


    def prepare_data(self):
        pass
        # call this once to convert val_data to memfile for saving memory
        # print(os.path.join(self.data_root, self.val_data_path, 'agedb_30', 'memfile'))
        # # if not os.path.isdir(os.path.join(self.data_root, self.val_data_path, 'agedb_30', 'memfile')):
        # print('making validation data memfile')
        # evaluate_utils.get_val_data(os.path.join(self.data_root, self.val_data_path))

        # if not os.path.isfile(self.concat_mem_file_name):
        #     # create a concat memfile
        #     concat = []
        #     for key in ['agedb_30', 'cfp_fp', 'lfw', 'cplfw', 'calfw']:
        #         np_array, issame = evaluate_utils.get_val_pair(path=os.path.join(self.data_root, self.val_data_path),
        #                                                        name=key,
        #                                                        use_memfile=False)
        #         concat.append(np_array)
        #     concat = np.concatenate(concat)
        #     evaluate_utils.make_memmap(self.concat_mem_file_name, concat)


    def setup(self, stage=None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == 'fit' or stage is None:
            print('creating train dataset')
            self.train_dataset = train_dataset(self.data_root,
                                               self.train_data_path,
                                               self.low_res_augmentation_prob,
                                               self.crop_augmentation_prob,
                                               self.photometric_augmentation_prob,
                                               self.swap_color_channel,
                                               self.use_mxrecord,
                                               self.output_dir
                                               )

            if 'faces_emore' in self.train_data_path and self.train_data_subset:
                # subset ms1mv2 dataset for reproducing the same setup in AdaFace ablation experiments.
                with open('assets/ms1mv2_train_subset_index.txt', 'r') as f:
                    subset_index = [int(i) for i in f.read().split(',')]
                    self.subset_ms1mv2_dataset(subset_index)

            print('creating val dataset')
            self.val_dataset = val_dataset(self.data_root, self.val_data_path, self.concat_mem_file_name)

        # Assign Test split(s) for use in Dataloaders
        if stage == 'test' or stage is None:
            self.test_dataset = test_dataset(self.data_root, self.val_data_path, self.concat_mem_file_name)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def subset_ms1mv2_dataset(self, subset_index):
        # remove too few example identites
        self.train_dataset.samples = [self.train_dataset.samples[idx] for idx in subset_index]
        self.train_dataset.targets = [self.train_dataset.targets[idx] for idx in subset_index]
        value_counts = pd.Series(self.train_dataset.targets).value_counts()
        to_erase_label = value_counts[value_counts<5].index
        e_idx = [i in to_erase_label for i in self.train_dataset.targets]
        self.train_dataset.samples = [i for i, erase in zip(self.train_dataset.samples, e_idx) if not erase]
        self.train_dataset.targets = [i for i, erase in zip(self.train_dataset.targets, e_idx) if not erase]

        # label adjust
        max_label = np.max(self.train_dataset.targets)
        adjuster = {}
        new = 0
        for orig in range(max_label+1):
            if orig in to_erase_label:
                continue
            adjuster[orig] = new
            new += 1

        # readjust class_to_idx
        self.train_dataset.targets = [adjuster[orig] for orig in self.train_dataset.targets]
        self.train_dataset.samples = [(sample[0], adjuster[sample[1]]) for sample in self.train_dataset.samples]
        new_class_to_idx = {}
        for label_str, label_int in self.train_dataset.class_to_idx.items():
            if label_int in to_erase_label:
                continue
            else:
                new_class_to_idx[label_str] = adjuster[label_int]
        self.train_dataset.class_to_idx = new_class_to_idx


def train_dataset(data_root, train_data_path,
                  low_res_augmentation_prob,
                  crop_augmentation_prob,
                  photometric_augmentation_prob,
                  swap_color_channel,
                  use_mxrecord,
                  output_dir):

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize((112, 112))
    ])

    if use_mxrecord:
        train_dir = os.path.join(data_root, train_data_path)
        train_dataset = AugmentRecordDataset(root_dir=train_dir,
                                             transform=train_transform,
                                             low_res_augmentation_prob=low_res_augmentation_prob,
                                             crop_augmentation_prob=crop_augmentation_prob,
                                             photometric_augmentation_prob=photometric_augmentation_prob,
                                             swap_color_channel=swap_color_channel,
                                             output_dir=output_dir)
    else:
        print("Train data path")
        if train_data_path == "mnist":
            train_dataset = torchvision.datasets.MNIST(
                root="./data", train=True, download=True, transform=train_transform
            )
        elif train_data_path == "cifar100":
            train_dataset = torchvision.datasets.CIFAR100(
                root="./data", train=True, download=True, transform=train_transform)
        else:
            train_dir = os.path.join(data_root, train_data_path, 'images')
            train_dataset = CustomImageFolderDataset(root=train_dir,
                                                    transform=train_transform,
                                                    low_res_augmentation_prob=low_res_augmentation_prob,
                                                    crop_augmentation_prob=crop_augmentation_prob,
                                                    photometric_augmentation_prob=photometric_augmentation_prob,
                                                    swap_color_channel=swap_color_channel,
                                                    output_dir=output_dir
                                                    )
            
    train_dataset = IndexedDatasetWrapper(train_dataset)

    return train_dataset


def val_dataset(data_root, val_data_path, concat_mem_file_name):
    val_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize((112, 112))
    ])
    # print("Coming in here")
    if val_data_path == "mnist":
        val_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=val_transform)
    elif val_data_path == "cifar100":
        val_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=val_transform)
    else:
        train_dir = os.path.join(data_root, val_data_path, 'images')
        val_dataset = CustomImageFolderDataset(root=train_dir,
                                                transform=val_transform)
        
    val_dataset = IndexedDatasetWrapper(val_dataset)

    return val_dataset


def test_dataset(data_root, val_data_path, concat_mem_file_name):
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112))
    ])

    if val_data_path == "mnist":
        val_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=test_transform)
    elif val_data_path == "cifar100":
        val_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=test_transform)
    else:
        train_dir = os.path.join(data_root, val_data_path, 'images')
        val_dataset = CustomImageFolderDataset(root=train_dir, transform=test_transform)
        
    val_dataset = IndexedDatasetWrapper(val_dataset)

    return val_dataset
