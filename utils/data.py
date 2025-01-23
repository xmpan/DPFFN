import torch.utils.data as Data
import torch
import os
import numpy as np
import scipy.io as scio
from glob import glob
from concurrent.futures import ThreadPoolExecutor

class MyDataSetVH(Data.Dataset):
    def __init__(self, eh_data, ev_data, label):
        super(MyDataSetVH, self).__init__()
        self.eh_data = eh_data
        self.ev_data = ev_data
        self.label = label

    def __len__(self):
        return self.eh_data.shape[0]

    def __getitem__(self, idx):
        return self.eh_data[idx, :, :], self.ev_data[idx, :, :], self.label[idx]

def sort_by_number(file_path):
    num = int(os.path.splitext(os.path.basename(file_path).split('=')[1])[0])
    return num

def load_mat_file(file_path):
    """Helper function to load data from a .mat file."""
    return scio.loadmat(file_path)['abs_RP']

def read_data_vh(folder_path1, folder_path2):
    """Read HRRP """
    # Define the shape and pre-allocate memory
    num_folders = len(os.listdir(folder_path1))
    num_files = len(os.listdir(os.path.join(folder_path1, os.listdir(folder_path1)[0])))
    data_shape = (num_folders * num_files, 512, 401)
    data_h = np.zeros(data_shape, dtype='float32')
    data_v = np.zeros(data_shape, dtype='float32')
    label = np.zeros(num_folders * num_files, dtype='int')

    # Process files in folder_path1 using parallel loading
    folder_lists = sorted(os.listdir(folder_path1), key=int)
    with ThreadPoolExecutor() as executor:
        for folder_index, folder_name in enumerate(folder_lists):
            label_value = int(folder_name) - 1
            inner_path = os.path.join(folder_path1, folder_name)
            file_paths = sorted(glob(os.path.join(inner_path, "*.mat")), key=sort_by_number)

            # Load files in parallel
            data_chunk = list(executor.map(load_mat_file, file_paths))
            for file_index, data in enumerate(data_chunk):
                full_index = folder_index * num_files + file_index
                data_h[full_index, :, :] = data
                label[full_index] = label_value

    # Process files in folder_path2 using parallel loading
    folder_lists_2 = sorted(os.listdir(folder_path2), key=int)
    with ThreadPoolExecutor() as executor:
        for folder_index, folder_name in enumerate(folder_lists_2):
            inner_path = os.path.join(folder_path2, folder_name)
            file_paths = sorted(glob(os.path.join(inner_path, "*.mat")), key=sort_by_number)

            # Load files in parallel
            data_chunk = list(executor.map(load_mat_file, file_paths))
            for file_index, data in enumerate(data_chunk):
                full_index = folder_index * num_files + file_index
                data_v[full_index, :, :] = data

    # Convert to PyTorch tensors
    data_h = torch.tensor(data_h, dtype=torch.float32)
    data_v = torch.tensor(data_v, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.long)

    return data_h, data_v, label
