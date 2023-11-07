import os
import math
import torch
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from torch.utils.data.dataset import Dataset

from datasets.utils.normals import normals_multi
from datasets.utils.normalize import normalize
from datasets.utils.optical_flow import dense_flow

def search_str_in_list(l,string):
    """
    This function takes a list l with strings inside 
    In the strings, we will search another string
    Return a new list with all the element from the original list where the string to search is included
    It return an empty list if string is not found in the elements of l
    """
    i = 0
    new_list = []
    while i<len(l):
        e = l[i]
        if string in e:
            new_list.append(e)
        i+=1
    return new_list

def get_good_numbers_of_frames(l,number_of_frames,skip_frames = 1):
    if skip_frames==1:
        center_of_list = math.floor(len(l) / 2)
        crop_limit = math.floor(number_of_frames / 2)
        start = center_of_list - crop_limit
        end = center_of_list + crop_limit
        l_croped = l[start: end + 1 if number_of_frames % 2 == 1 else end]
    else:
        i = 0
        l_croped = []
        while i<len(l) and len(l_croped)<number_of_frames:
            path = l[i]
            l_croped.append(path)
            i+=skip_frames
    return l_croped

class SHREC(Dataset):
    def __init__(self,path,data_type,split="train",number_of_labels = 14,transforms=None, n_frames=30, optical_flow=False,imsize=(224,224)):
        """Constructor method for Briareo Dataset class

        Args:
            split (str, optional): Current procedure phase (train, test, val)
            data_type (str, optional): Input data type (depth, rgb, normals, ir)
            transform (Object, optional): Data augmentation transformation for every data
            n_frames (int, optional): Number of frames selected for every input clip
            optical_flow (bool, optional): Useless it's only for compatibilty with Briareo Dataset
            imsize (tuple,optional): Size of image 

        """
        super().__init__()

        self.dataset_path = Path(path)
        self.split = split
        self.data_type = data_type
        self.imsize = imsize
        self.transforms = transforms
        self.n_frames = n_frames
        if number_of_labels==14:
            self.number_of_labels = number_of_labels
        else:
            self.number_of_labels = 28

        print("Loading SHREC {} dataset...".format(split.upper()), end=" ")

        split = self.split 
        if split == "val":
            split = "test"
        
        self.data = pd.read_csv(self.dataset_path /"{}.csv".format(self.split))
        
        self.Allpaths = self.data["path"]
        self.labels = self.data["labels_"+str(number_of_labels)]
        print("done.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.Allpaths[idx]
        label = self.labels[idx]

        clip = []
       
        
        dirs_in_path = os.listdir(str(self.dataset_path / Path(path)))
        frames_paths = search_str_in_list(dirs_in_path,"depth")
        if len(frames_paths)<self.n_frames:
            pass
        else:
            frames_paths = get_good_numbers_of_frames(frames_paths,self.n_frames,len(frames_paths)//self.n_frames)
        i=0
        while i<len(frames_paths):
            path_frame = frames_paths[i]
            path_to_open = str(self.dataset_path / Path(path) / Path(path_frame))
            img = cv2.imread(path_to_open,cv2.IMREAD_ANYDEPTH)
            img = cv2.resize(img,self.imsize)
            img = np.expand_dims(img, axis=2)
            clip.append(img)
            i+=1
        
        if len(frames_paths)<self.n_frames:
            misses_frames = self.n_frames-len(frames_paths)
            #print("In the {} elements missing {} frames".format(idx,misses_frames))
            i = 0
            last_frame = clip[-1]
            while i<misses_frames:
                clip.append(last_frame)
                i+=1
            
        clip = np.array(clip)
        clip = clip.transpose(1, 2, 3, 0)
        if self.data_type in ["normal", "normals"]:
            clip = normals_multi(clip)
        else:
            clip = normalize(clip)

        if self.transforms is not None:
            aug_det = self.transforms.to_deterministic()
            clip = np.array([aug_det.augment_image(clip[..., i]) for i in range(clip.shape[-1])]).transpose(1, 2, 3, 0)
            #clip = np.array([aug_det.augment_image(clip[..., i]) for i in range(clip.shape[-1])])
            
        clip = torch.from_numpy(clip.reshape(clip.shape[0], clip.shape[1],-1).transpose(2, 0, 1))
        label = torch.LongTensor(np.asarray([label]))
        
        return clip.float(), label
