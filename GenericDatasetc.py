import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from torchvision.transforms import ToTensor
import numpy as np
import os
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root=self.data_dir, transform=self.transform)
    def __getitem__(self, idx):
        paths = self.dataset.imgs[idx][0]  # Get file path associated with the image
        return self.dataset[idx], paths
    def __len__(self):
        return len(self.dataset)


class CustomCSVDataset(Dataset):
    def __init__(self, csv_file = None,data_frame = None, data_dir = '', transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if data_frame is None:
            self.annotations = pd.read_csv(csv_file)
        else:
            self.annotations = data_frame
        self.root_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.annotations.iloc[idx, 3]

        if self.transform:
            image = self.transform(image)

        sample = (image, label)
        return image, label

class GenericDatasetLoader:
    def __init__(self, dataset_name = None, root_dir='', transform = None, batch_size=1, shuffle=True, csv_file = None, data_frame = None,**kwargs):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.transform = transform
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.csv_file = csv_file
        self.data_frame = data_frame
        self.kwargs = kwargs

    def load_dataset(self, split='train'):
        if self.dataset_name == 'CIFAR10':
            dataset = self.load_cifar(split)
        elif self.dataset_name == 'MNIST':
            dataset = self.load_mnist(split)
        elif self.dataset_name == 'CUSTOM':
            dataset = self.load_custom(split)
        else:
            raise ValueError("Unsupported dataset: {}".format(self.dataset_name))


        return dataset

    def load_cifar(self, split='train'):
        if split == 'train':
            dataset = datasets.CIFAR10(root=self.root_dir, train=True, transform=self.transform, download=True)
        else:
            dataset = datasets.CIFAR10(root=self.root_dir, train=False, transform=self.transform, download=True)

        return dataset

    def load_mnist(self, split='train'):
        if split == 'train':
            dataset = datasets.MNIST(root=self.root_dir, train=True, transform=self.transform, download=True)
        else:
            dataset = datasets.MNIST(root=self.root_dir, train=False, transform=self.transform, download=True)

        return dataset

    def load_custom(self, split='train'):
        if self.csv_file is None and self.data_frame is None:
            if split == 'train':
                dataset = CustomDataset(data_dir=self.root_dir + '/train', transform=self.transform)
            else:
                dataset = CustomDataset(data_dir=self.root_dir + '/test', transform=self.transform)
                # dataset = CustomSVMDataSet(data_dir=self.root_dir + '/test', transform=self.transform)
            return dataset
        else:
            dataset = CustomCSVDataset(csv_file=self.csv_file,data_frame=self.data_frame, transform=self.transform)
            return dataset

    def create_dataloader(self, split='train'):
        dataset = self.load_dataset(split)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        return loader



    # def dataloader_to_dataframe(self,dataloader):
    #     data = []
    #     for batch in dataloader:
    #         # Assuming batch is a tuple (input_data, target)
    #         input_data, target = batch[0]
    #         # Flatten the input_data tensor
    #         input_data = input_data.view(input_data.size(0), -1)
    #         # Convert tensor to numpy array and then to list
    #         input_data = input_data.numpy()
    #         target = target.numpy().tolist()
    #         # Append data from batch to list
    #         data.extend(zip(input_data, target))
    #     # Create DataFrame from list of tuples
    #     df = pd.DataFrame(data, columns=['input_data', 'target'])
    #     return df

    def dataloader_to_dataframe(self,dataloader):
        data = []
        save_file = []
        for batch in dataloader:
            # Assuming batch is a tuple (input_data, target)
            path = batch[1]
            input_data, target = batch[0]
            # Flatten the input_data tensor
            input_data = input_data.view(input_data.size(0), -1)
            # Convert tensor to numpy array
            input_data = input_data.numpy()
            target = target.numpy()
            # Append data from batch to list
            data.append((input_data, target))
            save_file.append((path[0],target[0]))
        # Convert list of tuples to a single NumPy array
        input_data_array = np.concatenate([item[0] for item in data], axis=0)
        target_array = np.concatenate([item[1] for item in data], axis=0)

        df = pd.DataFrame(save_file, columns=['File Path','True Label'])

        df.to_csv('SVM_abnormal.csv', index=False)
        return input_data_array, target_array

