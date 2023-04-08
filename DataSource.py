import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import cv2
import torch.utils.data as data

class Datasource(data.Dataset):
    def __init__(self, root, resize=224, crop_size=224, train=True):
        self.root = os.path.expanduser(root)
        self.resize = resize
        self.crop_size = crop_size
        self.train = train
        
        self.images_path = []
        self.labels = []
        if train:
            train_path = root + '\\train'
            #print(train_path)
            for roots, files, dir in os.walk(train_path, topdown=False):
                    for name in dir:
                        full = os.path.join(roots, name)
                        if full.find("jpg") != -1:
                            self.images_path.append(full)
                        if full.find("angry")  !=-1:
                            self.labels.append(0)
                        if full.find("disgust")  !=-1:
                            self.labels.append(1)
                        if full.find("fear")  !=-1:
                            self.labels.append(2)
                        if full.find("happy")  !=-1:
                            self.labels.append(3)
                        if full.find("sad")  !=-1:
                            self.labels.append(4)
                        if full.find("surprise")  !=-1:
                            self.labels.append(5)
                        if full.find("neutral")  !=-1:
                            self.labels.append(6)

        
        else:
            test_path = root + '\\test'
            for roots, files, dir in os.walk(test_path, topdown=False):
                for name in dir:
                    full = os.path.join(roots, name)
                    if full.find("jpg") != -1:
                        self.images_path.append(full)
                        if full.find("angry")  !=-1:
                            self.labels.append(0)
                        if full.find("disgust")  !=-1:
                            self.labels.append(1)
                        if full.find("fear")  !=-1:
                            self.labels.append(2)
                        if full.find("happy")  !=-1:
                            self.labels.append(3)
                        if full.find("sad")  !=-1:
                            self.labels.append(4)
                        if full.find("surprise")  !=-1:
                            self.labels.append(5)
                        if full.find("neutral")  !=-1:
                            self.labels.append(6)


        # TODO: Define preprocessing

    def resized(self, data):
        # TODO: Perform preprocessing
        comp = T.ToTensor()
        data = comp(data)
        if self.train:
            compose = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(),T.RandomVerticalFlip(), T.Normalize(mean=0.5, std=0.5)])
            data = compose(data)
        else:
            compose = T.Compose([T.Resize((224, 224))])
            data = compose(data)
        return data

    def __getitem__(self, index):
        """
        return the data of one image
        """
        img_path = self.images_path[index]
        data = Image.open(img_path)
        # TODO: Perform preprocessing
        data = self.resized(data)
        cv2.imwrite("data_check.jpg", np.asarray(torch.permute(data, (1, 2, 0)) * 255))
        #print("Data saved!")
        #print(data.shape)
        return data, self.labels[index]

    def __len__(self):
        return len(self.images_path)