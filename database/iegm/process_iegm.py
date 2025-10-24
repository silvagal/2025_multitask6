import csv
import cv2
import numpy as np
import pandas as pd
import os
import sys
import torch


from collections import OrderedDict
from ecgdetectors import Detectors
from time import sleep
from tqdm import tqdm
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

SIZE=1250
BATCH_SIZE = 256

def loadCSV(csvf):
    """
    return a dict saving the information of csv
    :param splitFile: csv file name
    :return: {label:[file1, file2 ...]}
    """
    dictLabels = {}
    with open(csvf) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader, None)  # skip (filename, label)
        for i, row in enumerate(csvreader):
            filename = row[0]
            label = row[1]

            # append filename to current label
            if label in dictLabels.keys():
                dictLabels[label].append(filename)
            else:
                dictLabels[label] = [filename]
    return dictLabels


def txt_to_numpy(filename, row):
    file = open(filename)
    lines = file.readlines()
    datamat = np.arange(row, dtype=np.cfloat)
    row_count = 0
    for line in lines:
        line = line.strip().split(' ')
        datamat[row_count] = line[0]
        row_count += 1

    return datamat


class ToTensor(object):
    def __call__(self, sample):
        text = sample['IEGM_seg']
        return {
            'IEGM_seg': torch.from_numpy(text),
            'label': sample['label']
        }


class IEGM_DataSET():
    def __init__(self, root_dir, indice_dir, mode, size, transform=None):
        self.root_dir = root_dir
        self.indice_dir = indice_dir
        self.size = size
        self.names_list = []
        self.transform = transform
        self.detectors = Detectors(250)

        csvdata_all = loadCSV(os.path.join(self.indice_dir, mode + '_indice.csv'))
        for i, (k, v) in enumerate(csvdata_all.items()):
          if os.path.isfile(os.path.join(self.root_dir, k)):
            self.names_list.append(str(k) + ' ' + str(v[0]))

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, idx):
        text_path = os.path.join(self.root_dir, self.names_list[idx].split(' ')[0])

        if not os.path.isfile(text_path):
            print(text_path + ' does not exist')
            return None

        IEGM_seg = txt_to_numpy(text_path, self.size).reshape(1, self.size, 1)
        label = int(self.names_list[idx].split(' ')[1])

        sample = {'IEGM_seg': IEGM_seg, 'ac': label}

        return sample
    

def save_train_data(set,str):
    print(f"Salvando sinais e rótulos do {str}set...")

    signals = []
    labels = []

    for i in tqdm(range(len(set)), desc="Salvando"):
        sample = set[i]
        if sample is None:
            continue

        signals.append(torch.tensor(sample['IEGM_seg'], dtype=torch.float32))
        labels.append(torch.tensor(sample['ac'], dtype=torch.long))

    signals_tensor = torch.stack(signals)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    os.makedirs("2025_multitask/database", exist_ok=True)

    torch.save(signals_tensor, f"2025_multitask/database/{str}_signals.pt")
    torch.save(labels_tensor, f"2025_multitask/database/{str}_labels.pt")

    print("Arquivos salvos em '2025_multitask/database/'.")



def main():
    #####################################################
    # READING DATA
    #####################################################
    print('starting....')
    path_data = "/media/work/guilhermesilva/datasets/tinyml_contest_data_training/"
    path_indices = "2025_multitask/data_indices"

    # Start dataset loading
    print('loading dataset.....')
    trainset = IEGM_DataSET(root_dir=path_data,
                            indice_dir=path_indices,
                            mode='train',
                            size=SIZE,
                            transform=transforms.Compose([ToTensor()]))

    save_train_data(trainset, "train")

    testset = IEGM_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='test',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))

    save_train_data(testset, "test")

    print("Training and Testing Dataset loading finish.")

if __name__ == '__main__':
    main()