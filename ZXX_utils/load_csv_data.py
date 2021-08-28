
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

def default_loader(path):
    return Image.open(path).convert('RGB')

class csv_Dataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['label']))
        self.imgs = imgs
        #print(len(self.imgs))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        filename = filename.replace('../', '/content/datasets/')
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, label-1

    def __len__(self):
        return len(self.imgs)

# generate triplet example
class csv_triplet_Dataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        label_dict = {}
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['label']))
            label = row['label']
            if str(label) not in label_dict.keys():
                label_dict[str(label)] = []
                label_dict[str(label)].append(index)
            else:
                label_dict[str(label)].append(index)
        self.imgs = imgs
        self.label_dict = label_dict
        #print(len(self.imgs))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        pos_index = index
        while pos_index==index:
            random_index = np.random.randint(0, len(self.label_dict[str(label)]))
            pos_index = self.label_dict[str(label)][random_index]
        neg_label = label
        while neg_label == label:
            neg_label = np.random.randint(0, len(self.label_dict))+1
        random_index = np.random.randint(0, len(self.label_dict[str(neg_label)]))
        neg_index = self.label_dict[str(neg_label)][random_index]
        pos_filename, _ = self.imgs[pos_index]
        neg_filename, _ = self.imgs[neg_index]

        img = self.loader(filename)
        pos_img = self.loader(pos_filename)
        neg_img = self.loader(neg_filename)
        if self.transform is not None:
            img = self.transform(img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

        return img, label-1, pos_img, neg_img

    def __len__(self):
        return len(self.imgs)


class csv_pair_Dataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['label']))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)

        return img, label-1

class csv_negative_Dataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        img_list = np.zeros([200,1])
        img_number = np.zeros([200,1])
        img_label = 1
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['label'], row['correct'],
                         row['top1'], row['top2'], row['top3'],
                         row['top4'], row['top5']))
            img_number[row['label']-1,0] +=1
            if row['label'] == img_label:
                img_list[img_label - 1] = index
                img_label +=1

        self.imgs = imgs
        #print(len(self.imgs))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.img_list = np.uint16(img_list)
        self.img_number = np.uint16(img_number)

    def __getitem__(self, index):
        filename, label, compute_stage, N_lable1, N_label2, N_label3, N_label4, N_label5 = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        tmp = np.random.randint(0, 20)
        if tmp<=5:
            N_label = N_lable1
        elif tmp<=10:
            N_label = N_label2
        elif tmp<=14:
            N_label = N_label3
        elif tmp<=17:
            N_label = N_label4
        else:
            N_label = N_label5
        random_number = np.random.randint(0, self.img_number[N_label-1,0])
        n_filename, n_label, _, _, _, _, _, _ = self.imgs[self.img_list[N_label-1,0]+random_number]
        n_img = self.loader(n_filename)
        if self.transform is not None:
            n_img = self.transform(n_img)

        return img, label-1, compute_stage, n_img, n_label-1

    def __len__(self):
        return len(self.imgs)


class csv_Dataset_big(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['label'], row['biglabel']))
        self.imgs = imgs
        #print(len(self.imgs))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename, label, big_label = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, label-1, big_label-1

    def __len__(self):
        return len(self.imgs)
