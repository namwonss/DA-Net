import os
import random
import sys
import torch.utils.data as data
import torchvision.transforms as tfs
from PIL import Image
from torchvision.transforms import functional as FF
from metrics import *
from option import opt

sys.path.append('net')
sys.path.append('')
crop_size = 'whole_img'
if opt.crop:
    crop_size = opt.crop_size

class RS_Dataset(data.Dataset):
    #def __init__(self,path,train,size=crop_size,format='.png',hazy='cloud',GT='label'): # RICE
    #def __init__(self,path,train,size=crop_size,format='.png',hazy='input',GT='target'): # SATE 1K
    def __init__(self,path,train,size=crop_size,format='.png',hazy='hazy',GT='GT'): # RSID
        super(RS_Dataset,self).__init__()
        self.size=size
        print('crop size',size)
        self.train=train
        self.format=format
        self.haze_imgs_dir=os.listdir(os.path.join(path,hazy))
        self.haze_imgs=[os.path.join(path,hazy,img) for img in self.haze_imgs_dir]
        self.clear_dir=os.path.join(path,GT)
    def __getitem__(self, index):
        haze=Image.open(self.haze_imgs[index])
        if isinstance(self.size,int):
            while haze.size[0]<self.size or haze.size[1]<self.size :
                index=random.randint(0,20000)
                haze=Image.open(self.haze_imgs[index])
        img=self.haze_imgs[index]
        id=img.split('/')[-1].split('_')[0]
        #clear_name=id+self.format
        clear_name=id
        clear=Image.open(os.path.join(self.clear_dir,clear_name))
        clear=tfs.CenterCrop(haze.size[::-1])(clear)
        if not isinstance(self.size,str):
            i,j,h,w=tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
            haze=FF.crop(haze,i,j,h,w)
            clear=FF.crop(clear,i,j,h,w)
        haze,clear=self.augData(haze.convert("RGB") ,clear.convert("RGB") )
        return haze,clear
    def augData(self,data,target):
        if self.train:
            rand_hor=random.randint(0,1)
            rand_rot=random.randint(0,3)
            data=tfs.RandomHorizontalFlip(rand_hor)(data)
            target=tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data=FF.rotate(data,90*rand_rot)
                target=FF.rotate(target,90*rand_rot)
        data=tfs.ToTensor()(data)
        data=tfs.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])(data)
        target=tfs.ToTensor()(target)
        return  data ,target
    def __len__(self):
        return len(self.haze_imgs)