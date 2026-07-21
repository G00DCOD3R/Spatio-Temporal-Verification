import os
import pickle

from sklearn.preprocessing import OneHotEncoder, LabelEncoder  
import torch
from torch.utils import data
from PIL import Image
import numpy as np
import fnmatch


# Adapted from https://github.com/HHTseng/video-classification/blob/master/Conv3D/functions.py

## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()



# load UCF101 actions names
def load_file_names_and_labels(data_path: str | os.PathLike, action_name_path: str | os.PathLike) -> tuple[list, list]:
    with open(action_name_path, 'rb') as f:
        action_names = pickle.load(f)   # load UCF101 actions names

    # convert labels -> category
    le = LabelEncoder()
    le.fit(action_names)

    # show how many classes there are
    # list(le.classes_)

    # convert category -> 1-hot
    action_category = le.transform(action_names).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)


    actions = []
    fnames = os.listdir(data_path)

    all_names = []
    for f in fnames:
        loc1 = f.find('v_')
        loc2 = f.find('_g')
        actions.append(f[(loc1 + 2): loc2])

        all_names.append(f)


    # list all data files
    all_X_list = all_names              # all video file names
    all_y_list = labels2cat(le, actions)    # all video labels
    return (all_X_list, all_y_list)



# load subset of UCF101 actions names from the action set
def load_subset_file_names_and_labels(data_path: str | os.PathLike, action_set: set) -> tuple[list, list]:

    # convert labels -> category
    action_list = list(action_set)
    le = LabelEncoder()
    le.fit(action_list)

    # show how many classes there are
    # list(le.classes_)

    # convert category -> 1-hot
    action_category = le.transform(action_list).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(action_category)


    actions = []
    fnames = os.listdir(data_path)

    subset_names = []
    for f in fnames:
        loc1 = f.find('v_')
        loc2 = f.find('_g')
        
        action_name = f[(loc1 + 2): loc2]
        if(action_name in action_set):
            actions.append(f[(loc1 + 2): loc2])
            subset_names.append(f)


    # list all data files
    sub_X_list = subset_names              # all video file names
    sub_y_list = labels2cat(le, actions)    # all video labels
    return (sub_X_list, sub_y_list)


## ---------------------- Dataloaders ---------------------- ##
# for 3DCNN
class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, frames, transform=None, begin_frame=1, end_frame=29, skip_frame=1):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames
        self.begin_frame = begin_frame
        self.end_frame = end_frame
        self.skip_frame = skip_frame
        

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        begin_frame, end_frame = self.begin_frame, self.end_frame

        if self.frames is None:
            # Read all images in the folder and compare with start and end frame
            image_counts = len(fnmatch.filter(os.listdir(os.path.join(path, selected_folder)), '*.jpg')) 
            
            # if the number of images are significantly greater, then sample the middle frames
            if image_counts > (self.end_frame - self.begin_frame) * 2:
                # select the middle frames
                middle_frame = int(image_counts / 2)
                begin_frame = middle_frame - int((self.end_frame - self.begin_frame) / 2)
                end_frame = middle_frame + int((self.end_frame - self.begin_frame) / 2)
            
            frames = np.arange(begin_frame, end_frame, self.skip_frame)
            
        else:
            frames = self.frames
        
        for i in frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i))).convert('L')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform).unsqueeze_(0)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                             # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y


class Dataset_3DCNN_RGB(Dataset_3DCNN):
    '''
        Dataset extending Dataset_3DCNN with RGB images
    '''
    def __init__(self, data_path, folders, labels, frames, transform=None, begin_frame=1, end_frame=29, skip_frame=1, is_image_names_needed=False):
        super().__init__(data_path, folders, labels, frames, transform, begin_frame, end_frame, skip_frame)
        self.transform = transform
        self.begin_frame = begin_frame
        self.end_frame = end_frame
        self.skip_frame = skip_frame
        self.is_image_names_needed = is_image_names_needed

    def read_images(self, path, selected_folder, use_transform):
        X = []
        image_ids = []
        
        begin_frame, end_frame = self.begin_frame, self.end_frame

        if self.frames is None:
            # Read all images in the folder and compare with start and end frame
            image_counts = len(fnmatch.filter(os.listdir(os.path.join(path, selected_folder)), '*.jpg')) 
            
            # if the number of images are significantly greater, then sample the middle frames
            if image_counts > (self.end_frame - self.begin_frame) * 2:
                # select the middle frames
                middle_frame = int(image_counts / 2)
                begin_frame = middle_frame - int((self.end_frame - self.begin_frame) / 2)
                end_frame = middle_frame + int((self.end_frame - self.begin_frame) / 2)
            
            frames = np.arange(begin_frame, end_frame, self.skip_frame)
            
        else:
            frames = self.frames
        
        for i in frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i))).convert('RGB')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image) # image shape is (channels, width, height)
            image_ids.append(i)
        X = torch.stack(X, dim=1) # (channels, depth, width, height) - stack in the depth dimension
        image_ids = torch.LongTensor(image_ids) # image ids
        return X, image_ids
    
    
    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X, image_ids = self.read_images(self.data_path, folder, self.transform)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])# (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        if self.is_image_names_needed:
            return X, y, image_ids
        return X, y
