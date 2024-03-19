import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os
from os import listdir
from skimage import io
import imageio

import re
import chardet

class KITTI2012_train(Dataset):
    def __init__(self, dataset_dir, stereo_type='mccnn', max_disp=192.0, modal_type=None, use_cost=True):
        self.stereo_type = stereo_type
        self.max_disp = max_disp
        self.modal_type = modal_type
        self.use_cost = use_cost

        self.image_path = os.path.join(dataset_dir, 'colored_0')
        image_names = sorted(listdir(self.image_path))
        image_names = [x for x in image_names if x.find("10.png") != -1 if not x.startswith('.')] 
        
        self.disp_path = os.path.join(dataset_dir, 'disp_0', stereo_type)
        disp_names = sorted(listdir(self.disp_path))
        disp_names = [x for x in disp_names if x.find("10.png") != -1 if not x.startswith('.')]
        
        self.gt_disp_path = os.path.join(dataset_dir, 'disp_occ')
        gt_disp_names = sorted(listdir(self.gt_disp_path))
        gt_disp_names = [x for x in gt_disp_names if x.find("10.png") != -1 if not x.startswith('.')] 

        self.image_names = image_names[:20]
        self.disp_names = disp_names[:20]
        self.gt_disp_names = gt_disp_names[:20]
        
        if use_cost:
            self.cost_path = os.path.join(dataset_dir, 'cost', stereo_type)
            cost_names = sorted(listdir(self.cost_path)) 
            cost_names = [x for x in cost_names if x.find("10.npy") != -1 if not x.startswith('.')]
            self.cost_names = cost_names[:20]

        if modal_type is not None:
            self.modal_path = os.path.join(dataset_dir, modal_type, stereo_type)
            modal_names = sorted(listdir(self.modal_path))
            modal_names = [x for x in modal_names if x.find("10.png") != -1 if not x.startswith('.')]
            self.modal_names = modal_names[:20]
        
    def __len__(self):
        return 20

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        imag_name = os.path.join(self.image_path,self.image_names[idx])
        imag = io.imread(imag_name)
        imag = torch.Tensor(imag.astype(np.float32)/256.)
        imag = imag.transpose(0,1).transpose(0,2)
        
        ch, hei, wei = imag.size() 
        
        disp_name = os.path.join(self.disp_path,self.disp_names[idx])
        disp = io.imread(disp_name)
        disp = torch.Tensor(disp.astype(np.float32)/256.).unsqueeze(0)
       
        gt_disp_name = os.path.join(self.gt_disp_path,self.gt_disp_names[idx])
        gt_disp = io.imread(gt_disp_name)
        gt_disp = torch.Tensor(gt_disp.astype(np.float32)/256.).unsqueeze(0)
        gt_conf = (torch.abs(disp-gt_disp) <= 3).type(dtype=torch.float)
        gt_conf[gt_disp==0] = -1
                        
        new_hei = 128
        new_wei = 256
        
        top = np.random.randint(0,hei-new_hei)
        left = np.random.randint(0,wei-new_wei)

        disp = disp[:,top:top+new_hei,left:left+new_wei]/self.max_disp
        imag = imag[:,top:top+new_hei,left:left+new_wei]
        gt_conf = gt_conf[:,top:top+new_hei,left:left+new_wei]
        
        if self.use_cost:
            cost_name = os.path.join(self.cost_path,self.cost_names[idx])
            cost = np.load(cost_name)
            cost = torch.Tensor(cost)
            cost = cost[:,top:top+new_hei,left:left+new_wei]

        if self.modal_type is not None:
            modal_name = os.path.join(self.modal_path,self.modal_names[idx])
            modal = io.imread(modal_name)
            modal  = torch.Tensor(modal.astype(np.float32)/255.).unsqueeze(0)
            modal = modal[:,top:top+new_hei,left:left+new_wei]

        if (self.use_cost) and (self.modal_type is not None):
            return {'cost': cost, 'disp': disp, 'imag': imag, 'modal': modal, 'gt': gt_conf}
        elif (self.use_cost) and (self.modal_type is None):
            return {'cost': cost, 'disp': disp, 'imag': imag, 'gt': gt_conf}
        elif (not self.use_cost) and (self.modal_type is not None):
            return {'disp': disp, 'imag': imag, 'modal': modal, 'gt': gt_conf}
        elif (not self.use_cost) and (self.modal_type is None):
            return {'disp': disp, 'imag': imag, 'gt': gt_conf}

class KITTI2015_test(Dataset):
    def __init__(self, dataset_dir, stereo_type='mccnn', max_disp=192.0, modal_type=None, use_cost=True):
        self.stereo_type = stereo_type
        self.max_disp = max_disp
        self.modal_type = modal_type
        self.use_cost = use_cost

        self.image_path = os.path.join(dataset_dir, 'image_2')
        image_names = sorted(listdir(self.image_path))
        image_names = [x for x in image_names if x.find("10.png") != -1 if not x.startswith('.')] 
        
        self.disp_path = os.path.join(dataset_dir, 'disp_0', stereo_type)
        disp_names = sorted(listdir(self.disp_path))
        disp_names = [x for x in disp_names if x.find("10.png") != -1 if not x.startswith('.')] 
        
        self.gt_disp_path = os.path.join(dataset_dir, 'disp_occ_0')
        gt_disp_names = sorted(listdir(self.gt_disp_path))
        gt_disp_names = [x for x in gt_disp_names if x.find("10.png") != -1 if not x.startswith('.')] 

        self.image_names = image_names
        self.disp_names = disp_names
        self.gt_disp_names = gt_disp_names

        if use_cost:    
            self.cost_path = os.path.join(dataset_dir, 'cost', stereo_type)
            cost_names = sorted(listdir(self.cost_path))
            cost_names = [x for x in cost_names if x.find("10.npy") != -1 if not x.startswith('.')]
            self.cost_names = cost_names

        if modal_type is not None:
            self.modal_path = os.path.join(dataset_dir, modal_type, stereo_type)
            modal_names = sorted(listdir(self.modal_path))
            modal_names = [x for x in modal_names if x.find("10.png") != -1 if not x.startswith('.')]
            self.modal_names = modal_names

    def __len__(self):
        return 200

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        imag_name = os.path.join(self.image_path,self.image_names[idx])
        imag = io.imread(imag_name)
        imag = torch.Tensor(imag.astype(np.float32)/256.)
        imag = imag.transpose(0,1).transpose(0,2)
        
        ch, hei, wei = imag.size() 
        
        disp_name = os.path.join(self.disp_path,self.disp_names[idx])
        disp = io.imread(disp_name)
        disp = torch.Tensor(disp.astype(np.float32)/256.).unsqueeze(0)
       
        gt_disp_name = os.path.join(self.gt_disp_path,self.gt_disp_names[idx])
        gt_disp = io.imread(gt_disp_name)
        gt_disp = torch.Tensor(gt_disp.astype(np.float32)/256.).unsqueeze(0)
        
        if self.use_cost:
            cost_name = os.path.join(self.cost_path,self.cost_names[idx])
            cost = np.load(cost_name)
            cost = torch.Tensor(cost)

        if self.modal_type is not None:
            modal_name = os.path.join(self.modal_path,self.modal_names[idx])
            modal = io.imread(modal_name)
            modal = torch.Tensor(modal.astype(np.float32)/255.).unsqueeze(0)

        if (self.use_cost) and (self.modal_type is not None):
            return {'cost': cost, 'disp': disp, 'gt_disp': gt_disp, 'imag': imag, 'modal': modal}
        elif (self.use_cost) and (self.modal_type is None):
            return {'cost': cost, 'disp': disp, 'gt_disp': gt_disp, 'imag': imag}
        elif (not self.use_cost) and (self.modal_type is not None):
            return {'disp': disp, 'gt_disp': gt_disp, 'imag': imag, 'modal': modal}
        elif (not self.use_cost) and (self.modal_type is None):
            return {'disp': disp, 'gt_disp': gt_disp, 'imag': imag}

class KITTI2012_test(Dataset):
    def __init__(self, dataset_dir, stereo_type='mccnn', max_disp=192.0, modal_type=None, use_cost=True):
        self.stereo_type = stereo_type
        self.max_disp = max_disp
        self.modal_type = modal_type
        self.use_cost = use_cost

        self.image_path = os.path.join(dataset_dir, 'colored_0')
        image_names = sorted(listdir(self.image_path))
        image_names = [x for x in image_names if x.find("10.png") != -1 if not x.startswith('.')] 
        
        self.disp_path = os.path.join(dataset_dir, 'disp_0', stereo_type)
        disp_names = sorted(listdir(self.disp_path))
        disp_names = [x for x in disp_names if x.find("10.png") != -1 if not x.startswith('.')] 
        
        self.gt_disp_path = os.path.join(dataset_dir, 'disp_occ')
        gt_disp_names = sorted(listdir(self.gt_disp_path))
        gt_disp_names = [x for x in gt_disp_names if x.find("10.png") != -1 if not x.startswith('.')] 

        self.image_names = image_names[20:]
        self.disp_names = disp_names[20:]
        self.gt_disp_names = gt_disp_names[20:]

        if use_cost:    
            self.cost_path = os.path.join(dataset_dir, 'cost', stereo_type)
            cost_names = sorted(listdir(self.cost_path))
            cost_names = [x for x in cost_names if x.find("10.npy") != -1 if not x.startswith('.')]
            self.cost_names = cost_names[20:]

        if modal_type is not None:
            self.modal_path = os.path.join(dataset_dir, modal_type, stereo_type)
            modal_names = sorted(listdir(self.modal_path))
            modal_names = [x for x in modal_names if x.find("10.png") != -1 if not x.startswith('.')]
            self.modal_names = modal_names[20:]

    def __len__(self):
        return 174

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        imag_name = os.path.join(self.image_path,self.image_names[idx])
        imag = io.imread(imag_name)
        imag = torch.Tensor(imag.astype(np.float32)/256.)
        imag = imag.transpose(0,1).transpose(0,2)
        
        ch, hei, wei = imag.size() 
        
        disp_name = os.path.join(self.disp_path,self.disp_names[idx])
        disp = io.imread(disp_name)
        disp = torch.Tensor(disp.astype(np.float32)/256.).unsqueeze(0)
       
        gt_disp_name = os.path.join(self.gt_disp_path,self.gt_disp_names[idx])
        gt_disp = io.imread(gt_disp_name)
        gt_disp = torch.Tensor(gt_disp.astype(np.float32)/256.).unsqueeze(0)
        
        if self.use_cost:
            cost_name = os.path.join(self.cost_path,self.cost_names[idx])
            cost = np.load(cost_name)
            cost = torch.Tensor(cost)

        if self.modal_type is not None:
            modal_name = os.path.join(self.modal_path,self.modal_names[idx])
            modal = io.imread(modal_name)
            modal = torch.Tensor(modal.astype(np.float32)/255.).unsqueeze(0)

        if (self.use_cost) and (self.modal_type is not None):
            return {'cost': cost, 'disp': disp, 'gt_disp': gt_disp, 'imag': imag, 'modal': modal}
        elif (self.use_cost) and (self.modal_type is None):
            return {'cost': cost, 'disp': disp, 'gt_disp': gt_disp, 'imag': imag}
        elif (not self.use_cost) and (self.modal_type is not None):
            return {'disp': disp, 'gt_disp': gt_disp, 'imag': imag, 'modal': modal}
        elif (not self.use_cost) and (self.modal_type is None):
            return {'disp': disp, 'gt_disp': gt_disp, 'imag': imag}

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    encode_type = chardet.detect(header)  
    header = header.decode(encode_type['encoding'])
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode(encode_type['encoding']))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip().decode(encode_type['encoding']))
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)

    return data, scale
        
class MID2014_test(Dataset):
    def __init__(self, dataset_dir, stereo_type='mccnn', max_disp=192.0, modal_type=None, use_cost=True):
        self.stereo_type = stereo_type
        self.max_disp = max_disp
        self.modal_type = modal_type
        self.use_cost = use_cost

        self.image_path = os.path.join(dataset_dir, 'im0')
        image_names = sorted(listdir(self.image_path))
        image_names = [x for x in image_names if x.find(".png") != -1 if not x.startswith('.')] 
        
        self.disp_path = os.path.join(dataset_dir, 'disp_0', stereo_type)
        disp_names = sorted(listdir(self.disp_path))
        disp_names = [x for x in disp_names if x.find(".png") != -1 if not x.startswith('.')] 
        
        self.gt_disp_path = os.path.join(dataset_dir, 'disp0GT')
        gt_disp_names = sorted(listdir(self.gt_disp_path))
        gt_disp_names = [x for x in gt_disp_names if x.find(".pfm") != -1 if not x.startswith('.')] 

        self.mask_nocc_path = os.path.join(dataset_dir, 'mask0nocc')
        mask_nocc_names = sorted(listdir(self.mask_nocc_path))
        mask_nocc_names = [x for x in mask_nocc_names if x.find(".png") != -1 if not x.startswith('.')] 

        self.image_names = image_names
        self.disp_names = disp_names
        self.gt_disp_names = gt_disp_names
        self.mask_nocc_names = mask_nocc_names

        if use_cost:    
            self.cost_path = os.path.join(dataset_dir, 'cost', stereo_type)
            cost_names = sorted(listdir(self.cost_path))
            cost_names = [x for x in cost_names if x.find(".npy") != -1 if not x.startswith('.')]
            self.cost_names = cost_names

        if modal_type is not None:
            self.modal_path = os.path.join(dataset_dir, modal_type, stereo_type)
            modal_names = sorted(listdir(self.modal_path))
            modal_names = [x for x in modal_names if x.find(".png") != -1 if not x.startswith('.')]
            self.modal_names = modal_names

    def __len__(self):
        return 15

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        imag_name = os.path.join(self.image_path,self.image_names[idx])
        imag = io.imread(imag_name)
        imag = torch.Tensor(imag.astype(np.float32)/256.)
        imag = imag.transpose(0,1).transpose(0,2)
        
        ch, hei, wei = imag.size() 
        
        disp_name = os.path.join(self.disp_path,self.disp_names[idx])
        disp = io.imread(disp_name)
        disp = torch.Tensor(disp.astype(np.float32)/256.).unsqueeze(0)
       
        gt_disp_name = os.path.join(self.gt_disp_path,self.gt_disp_names[idx])
        gt_disp, _ = readPFM(gt_disp_name)
        gt_disp[gt_disp == np.inf] = 0
        gt_disp = torch.Tensor(np.ascontiguousarray(gt_disp, dtype=np.float32)).unsqueeze(0)
        
        mask_nocc_name = os.path.join(self.mask_nocc_path,self.mask_nocc_names[idx])
        mask_nocc = imageio.imread(mask_nocc_name)
        valid = (mask_nocc == 255)
        valid = torch.Tensor(valid).unsqueeze(dim=0)

        if self.use_cost:
            cost_name = os.path.join(self.cost_path,self.cost_names[idx])
            cost = np.load(cost_name)
            cost = torch.Tensor(cost)

        if self.modal_type is not None:
            modal_name = os.path.join(self.modal_path,self.modal_names[idx])
            modal = io.imread(modal_name)
            modal = torch.Tensor(modal.astype(np.float32)/255.).unsqueeze(0)

        if (self.use_cost) and (self.modal_type is not None):
            return {'cost': cost, 'disp': disp, 'gt_disp': gt_disp, 'imag': imag, 'modal': modal, 'mask': valid}
        elif (self.use_cost) and (self.modal_type is None):
            return {'cost': cost, 'disp': disp, 'gt_disp': gt_disp, 'imag': imag, 'mask': valid}
        elif (not self.use_cost) and (self.modal_type is not None):
            return {'disp': disp, 'gt_disp': gt_disp, 'imag': imag, 'modal': modal, 'mask': valid}
        elif (not self.use_cost) and (self.modal_type is None):
            return {'disp': disp, 'gt_disp': gt_disp, 'imag': imag, 'mask': valid}