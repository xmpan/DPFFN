import torch.utils.data as Data
import torch
import os
import numpy as np
import scipy.io as scio

class MyDataSetVH(Data.Dataset):
    def __init__(self, eh_data,ev_data,label):
        super(MyDataSetVH, self).__init__()
        self.eh_data = eh_data
        self.ev_data = ev_data
        self.label = label

    def __len__(self):
        return self.eh_data.shape[0]

    def __getitem__(self, idx):
        return self.eh_data[idx,: ,:],self.ev_data[idx,: ,:], self.label[idx]
    
def sort_by_number(item):
    num = int(item.split('=')[1].split('.')[0])
    return num


def read_data_vh(folder_path1,folder_path2):
    folder_lists = os.listdir(folder_path1)
    folder_1 = os.listdir(os.path.join(folder_path1, folder_lists[0]))
    data_h = np.zeros([len(folder_1)*10, 512, 401], dtype='float')
    label = np.zeros(len(folder_1)*10)
    folder_lists.sort(key=lambda x:int(x))
    for folder_list_1 in folder_lists:
        Label = int(folder_list_1) - 1
        inner_path = os.path.join(folder_path1, folder_list_1)
        file_names = os.listdir(inner_path)
        count = 0
        file_names.sort(key=sort_by_number)
        base = int(file_names[0].split('=')[1].split('.')[0])-1
        for file_name in file_names:
            count = int(file_name.split('=')[1].split('.')[0]) - 1 - base
            file_path = os.path.join(inner_path, file_name)
            data_h[Label*len(file_names)+count, :, :] = scio.loadmat(file_path)['abs_RP']
            label[Label*len(file_names)+count] = int(folder_list_1)-1
             
    data_h = torch.Tensor(data_h)

    folder_lists_2 = os.listdir(folder_path2)
    folder_2 = os.listdir(os.path.join(folder_path2, folder_lists_2[0]))
    data_v = np.zeros([len(folder_2)*10, 512, 401], dtype='float')
    folder_lists_2.sort(key=lambda x:int(x))
    for folder_list_2 in folder_lists:
        Label = int(folder_list_2) - 1
        inner_path = os.path.join(folder_path2, folder_list_2) 
        file_names = os.listdir(inner_path)
        count = 0
        file_names.sort(key=sort_by_number)
        base = int(file_names[0].split('=')[1].split('.')[0])-1
        for file_name in file_names:
            count = int(file_name.split('=')[1].split('.')[0]) - 1 - base
            file_path = os.path.join(inner_path, file_name)
            data_v[Label*len(file_names)+count, :, :] = scio.loadmat(file_path)['abs_RP']
             
    data_v = torch.Tensor(data_v)
    label = torch.Tensor(label)
        
    return(data_h,data_v,label)