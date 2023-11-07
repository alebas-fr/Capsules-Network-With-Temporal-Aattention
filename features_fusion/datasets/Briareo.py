from multiprocessing import parent_process
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

from datasets.utils.utils_briareo import from_json_to_list


class Briareo(Dataset):
    """Briareo Dataset class"""
    def __init__(self, configer, path, split="train", data_type='depth',number_of_labels=12, transforms=None, n_frames=30, optical_flow=False):
        """Constructor method for Briareo Dataset class

        Args:
            configer (Configer): Configer object for current procedure phase (train, test, val)
            split (str, optional): Current procedure phase (train, test, val)
            data_type (str, optional): Input data type (depth, rgb, normals, ir)
            transform (Object, optional): Data augmentation transformation for every data
            n_frames (int, optional): Number of frames selected for every input clip
            optical_flow (bool, optional): Flag to choose if calculate optical flow or not

        """
        super().__init__()

        self.dataset_path = Path(path)
        self.split = split
        self.data_type = data_type
        self.optical_flow = optical_flow
        if self.data_type in ["normal", "normals"] and self.optical_flow:
            raise NotImplementedError("Optical flow for normals image is not supported.")

        self.transforms = transforms
        self.n_frames = n_frames if not optical_flow else n_frames + 1
        self.multiModal = False
        self.data = None

        print("Loading Briareo {} dataset...".format(split.upper()), end=" ")
        if isinstance(data_type,list):
            self.multiModal = True
            j=0
            fixed_data = list()
            while j<len(data_type):
                one_data_list = list()
                data_type_unique = data_type[j]
                if data_type_unique in ["normal", "normals"]:
                    data_type_unique = "depth"
                data = np.load(self.dataset_path / "splits" / (self.split if self.split != "val" else "train") /
                                            "{}_{}.npz".format(data_type_unique, self.split), allow_pickle=True)['arr_0']

                # Prepare clip for the selected number of frames n_frame
                for i, record in enumerate(data):
                    paths = record['data']

                    center_of_list = math.floor(len(paths) / 2)
                    crop_limit = math.floor(self.n_frames / 2)

                    start = center_of_list - crop_limit
                    end = center_of_list + crop_limit
                    paths_cropped = paths[start: end + 1 if self.n_frames % 2 == 1 else end]
                    if data_type_unique == 'leapmotion':
                        valid = np.array(record['valid'][start: end + 1 if self.n_frames % 2 == 1 else end])
                        if valid.sum() == len(valid):
                            data[i]['data'] = paths_cropped
                            one_data_list.append(data[i])
                    else:
                        data[i]['data'] = paths_cropped
                        one_data_list.append(data[i])
                fixed_data.append(one_data_list)
                j+=1
            data_np = np.array(fixed_data)
            print("We have {} paths".format(data_np.shape))
            self.data = pd.DataFrame(data_np.T,columns=data_type)
            self.data.to_csv(split+".csv")
            
        else:
            if data_type in ["normal", "normals"]:
                data_type = "depth"
            data = np.load(self.dataset_path / "splits" / (self.split if self.split != "val" else "train") /
                                        "{}_{}.npz".format(data_type, self.split), allow_pickle=True)['arr_0']

            # Prepare clip for the selected number of frames n_frame
            fixed_data = list()
            for i, record in enumerate(data):
                paths = record['data']

                center_of_list = math.floor(len(paths) / 2)
                crop_limit = math.floor(self.n_frames / 2)

                start = center_of_list - crop_limit
                end = center_of_list + crop_limit
                paths_cropped = paths[start: end + 1 if self.n_frames % 2 == 1 else end]
                if self.data_type == 'leapmotion':
                    valid = np.array(record['valid'][start: end + 1 if self.n_frames % 2 == 1 else end])
                    if valid.sum() == len(valid):
                        data[i]['data'] = paths_cropped
                        fixed_data.append(data[i])
                else:
                    data[i]['data'] = paths_cropped
                    fixed_data.append(data[i])
            self.data = np.array(fixed_data)
        print("done.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.multiModal:
            label = self.data[self.data_type[0]][idx]['label']
            paths = None
            j= 0 
            clip = None
            all_clip = list()
            while j<len(self.data_type):
                data_type_unique = self.data_type[j]
                paths = self.data[data_type_unique][idx]['data']
                clip_one_type = list()
                
                for p in paths:
                    if self.data_type == "leapmotion":
                        img = from_json_to_list(os.path.join(self.dataset_path, p))[0]
                    else:
                        if data_type_unique in ["depth", "normal", "normals"]:
                            img = np.load(str(self.dataset_path / p), allow_pickle=True)['arr_0']
                            if data_type_unique in ["normal", "normals"]:
                                img *= 1000
                        elif data_type_unique in ["ir"]:
                            img = cv2.imread(str(self.dataset_path / p), cv2.IMREAD_ANYDEPTH)
                        else:
                            img = cv2.imread(str(self.dataset_path / p), cv2.IMREAD_COLOR)
                        img = cv2.resize(img, (224, 224))
                        if data_type_unique != "rgb":
                            img = np.expand_dims(img, axis=2)
                    clip_one_type.append(img)
                clip_one_type = np.array(clip_one_type)
                clip_one_type = clip_one_type.transpose(1, 2, 3, 0)
                if self.data_type in ["normal", "normals"]:
                    clip_one_type = normals_multi(clip_one_type)
                else:
                    if self.optical_flow:
                        clip_one_type = dense_flow(clip_one_type, self.data_type == "rgb")
                    clip_one_type = normalize(clip_one_type)
                if self.transforms is not None:
                    aug_det = self.transforms.to_deterministic()
                    clip_one_type = np.array([aug_det.augment_image(clip_one_type[..., i]) for i in range(clip_one_type.shape[-1])]).transpose(1, 2, 3, 0)
                clip_one_type = torch.from_numpy(clip_one_type.transpose(3,2,0,1))
                #print(data_type_unique,"Shape of clip: ",clip_one_type.shape)
                all_clip.append(clip_one_type)
                j+=1
            j = 1
            clip = all_clip[0]
            while j<len(all_clip):
                clip = torch.cat((clip,all_clip[j]),1)
                j+=1

                

        else:
            paths = self.data[idx]['data']
            label = self.data[idx]['label']

            clip = list()
            for p in paths:
                if self.data_type == "leapmotion":
                    img = from_json_to_list(os.path.join(self.dataset_path, p))[0]
                else:
                    if self.data_type in ["depth", "normal", "normals"]:
                        img = np.load(str(self.dataset_path / p), allow_pickle=True)['arr_0']
                        if self.data_type in ["normal", "normals"]:
                            img *= 1000
                    elif self.data_type in ["ir"]:
                        img = cv2.imread(str(self.dataset_path / p), cv2.IMREAD_ANYDEPTH)
                    else:
                        img = cv2.imread(str(self.dataset_path / p), cv2.IMREAD_COLOR)
                    img = cv2.resize(img, (224, 224))
                    if self.data_type != "rgb":
                        img = np.expand_dims(img, axis=2)
                clip.append(img)

            clip = np.array(clip).transpose(1, 2, 3, 0)

            if self.data_type in ["normal", "normals"]:
                clip = normals_multi(clip)
            else:
                if self.optical_flow:
                    clip = dense_flow(clip, self.data_type == "rgb")
                clip = normalize(clip)

            if self.transforms is not None:
                aug_det = self.transforms.to_deterministic()
                clip = np.array([aug_det.augment_image(clip[..., i]) for i in range(clip.shape[-1])]).transpose(1, 2, 3, 0)

            clip = torch.from_numpy(clip.reshape(clip.shape[0], clip.shape[1], -1).transpose(2, 0, 1))
            label = torch.LongTensor(np.asarray([label]))
        return clip.float(), label
