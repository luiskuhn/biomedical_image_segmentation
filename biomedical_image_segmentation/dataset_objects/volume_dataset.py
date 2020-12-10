import numpy as np
import torch
from torch.utils.data import Dataset

import torchio
from torchio.transforms import RandomAffine


class VolumeDataset(Dataset):

    def __init__(self, ids, path="ds/", apply_trans=True, n_class=2):
        """
        Args:

        assumes the filenames of an image pair (input and label) are img_<ID>.pt and lab_<ID>.pt
            
        """

        self.n_class = n_class
                        
        self.inputs = []
        self.labels = []

        for sample in ids:
            
            #print('sample: ' + str(sample))
            
            #########################################

            vol = torch.load(path + "img_" + str(sample) + ".pt")
            label = torch.load(path + "lab_" + str(sample) + ".pt")

            #############
            #change type
            #vol = vol.astype(np.float32)
            label = label.astype(np.float32)

            vol = np.expand_dims(vol, axis=0) ##add channel dim

            vol = torch.tensor(vol)
            label = torch.tensor(label)
            #label = label.type(torch.long)

            self.inputs.append(vol)
            self.labels.append(label)

        ###################################
        #transforms

        self.apply_trans = apply_trans

        self.transform = RandomAffine(
            scales=(1.0, 1.0),
            degrees=(0, 2.0),
            translation=(0, 2.0),
            isotropic=True,
            default_pad_value='mean')#,
            #image_interpolation='bspline')

    def apply_transformation(self, vol, label):

        lab = label.unsqueeze(0)

        subject = torchio.Subject(
            img=torchio.ScalarImage(tensor=vol),
            lab=torchio.ScalarImage(tensor=lab))

        trans_subject = self.transform(subject)

        trans_img = trans_subject['img'].data
        trans_lab = torch.clamp(torch.round(trans_subject['lab'].data), min=0, max=(self.n_class -1) ) #binnary class

        trans_lab = trans_lab.squeeze(0)

        return (trans_img, trans_lab)


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):

        vol = self.inputs[idx]
        label = self.labels[idx]

        pair = (vol, label)

        ###########
        #transform
        if self.apply_trans:
            pair = self.apply_transformation(vol, label)
        ##########


        return pair
