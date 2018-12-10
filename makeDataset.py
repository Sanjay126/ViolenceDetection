import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2

class makeDataset(Dataset):
    def __init__(self, dataset, labels, spatial_transform, seqLen=20):
        self.spatial_transform = spatial_transform
        self.images = dataset
        self.labels = labels
        self.seqLen = seqLen

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        vid_name = self.images[idx]
        label = self.labels[idx]
        inpSeq = []
        self.spatial_transform.randomize_parameters()
        vid=cv2.VideoCapture(vid_name)
        index=0
        frame =None
        while(index<=self.seqLen):
            # Extract images
            ret, frame = vid.read()
            # end of frames
            if not ret: 
                break
            # next frame
            index+=1
            img=Image.fromarray(frame)
            inpSeq.append(self.spatial_transform(img.convert('RGB')))
        for i in range(3):
            k=inpSeq[0]-inpSeq[1]
            k.save("./diff"+self.images[idx]+str(i),".png")
        inpSeq = torch.stack(inpSeq, 0)
        return inpSeq, label
