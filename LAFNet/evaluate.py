import os
import cv2
import re
import chardet
import argparse
import numpy as np

import torch
import torch.optim
import torch.utils.data

import models

from os import listdir
from skimage import io
from tqdm import tqdm

def readPFM(file):
    file = open(file, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    encode_type = chardet.detect(header)  
    header = header.decode(encode_type["encoding"])
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode(encode_type["encoding"]))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip().decode(encode_type["encoding"]))
    if scale < 0: # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">" # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)

    return data, scale

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="../samples/kitti-stereo-2012", help="dataset directory")
parser.add_argument("--dataset_type", type=str, default="kitti2012", help="dataset directory path")
parser.add_argument("--stereo_type", type=str, default="psmnet", help="stereo-matching network type")
parser.add_argument("--max_disp", type=float, default=192.0, help="maximum disparity of stereo-matching network")
parser.add_argument("--modal_type", type=str, default=None, help="additional input modality type")
parser.add_argument("--use_cost", action="store_true", default=False)
parser.add_argument("--weights_path", type=str, default=None)
parser.add_argument("--result_dir", type=str, default="./results", help="result directory path")
opt = parser.parse_args()

os.makedirs(opt.result_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.LAFNet_CVPR2019(modal_type=opt.modal_type,
                               use_cost=opt.use_cost).to(device)
model.load_state_dict(torch.load(opt.weights_path))
model.eval()

if opt.dataset_type == "kitti2015":
    image_path = os.path.join(opt.dataset_dir, "image_2")
    image_names = sorted(listdir(image_path))
    image_names = [x for x in image_names if x.find("10.png") != -1 if not x.startswith(".")]

    gt_disp_path = os.path.join(opt.dataset_dir, "disp_occ_0")
    gt_disp_names = sorted(listdir(gt_disp_path))
    gt_disp_names = [x for x in gt_disp_names if x.find("10.png") != -1 if not x.startswith(".")]
elif opt.dataset_type == "kitti2012":
    image_path = os.path.join(opt.dataset_dir, "colored_0")
    image_names = sorted(listdir(image_path))
    image_names = [x for x in image_names if x.find("10.png") != -1 if not x.startswith(".")]

    gt_disp_path = os.path.join(opt.dataset_dir, "disp_occ")
    gt_disp_names = sorted(listdir(gt_disp_path))
    gt_disp_names = [x for x in gt_disp_names if x.find("10.png") != -1 if not x.startswith(".")]    
elif opt.dataset_type == "mid2014":
    image_path = os.path.join(opt.dataset_dir, "im0")
    image_names = sorted(listdir(image_path))
    image_names = [x for x in image_names if x.find(".png") != -1 if not x.startswith(".")]

    gt_disp_path = os.path.join(opt.dataset_dir, "disp0GT")
    gt_disp_names = sorted(listdir(gt_disp_path))
    gt_disp_names = [x for x in gt_disp_names if x.find(".pfm") != -1 if not x.startswith(".")]    

disp_path = os.path.join(opt.dataset_dir, "disp_0", opt.stereo_type)
disp_names = sorted(listdir(disp_path))
disp_names = [x for x in disp_names if x.find(".png") != -1 if not x.startswith(".")]

if opt.modal_type is not None:
    modal_path = os.path.join(opt.dataset_dir, opt.modal_type, opt.stereo_type)
    modal_names = sorted(listdir(modal_path))
    modal_names = [x for x in modal_names if x.find(".png") != -1 if not x.startswith(".")]

if opt.use_cost:
    cost_path = os.path.join(opt.dataset_dir, "cost", opt.stereo_type)
    cost_names = sorted(listdir(cost_path))
    cost_names = [x for x in cost_names if x.find(".npy") != -1 if not x.startswith(".")]

AUCs = []
opts = []
b3s = []

if opt.dataset_type == "kitti2012":
    image_names = image_names[20:]
    gt_disp_names = gt_disp_names[20:]
    disp_names = disp_names[20:]
    if opt.modal_type is not None:
        modal_names = modal_names[20:]
    if opt.use_cost:
        cost_names = cost_names[20:]

pbar = tqdm(total=len(image_names))

with torch.no_grad():
    for idx in range(len(image_names)):
        imag_name = os.path.join(image_path,image_names[idx])
        imag = io.imread(imag_name)
        imag = torch.Tensor(imag.astype(np.float32)/256.)
        imag = imag.transpose(0,1).transpose(0,2)

        ch, hei, wei = imag.size()

        disp_name = os.path.join(disp_path,disp_names[idx])
        disp = io.imread(disp_name)
        disp = torch.Tensor(disp.astype(np.float32)/256.).unsqueeze(0)
        
        if opt.dataset_type == "mid2014":
            gt_disp, _ = readPFM(os.path.join(gt_disp_path, gt_disp_names[idx]))
            gt_disp[gt_disp == np.inf] = 0
            gt_disp = np.ascontiguousarray(gt_disp, dtype=np.float32)
            gt_disp = torch.Tensor(gt_disp).unsqueeze(0)
        else:
            gt_disp_name = os.path.join(gt_disp_path,gt_disp_names[idx])
            gt_disp = io.imread(gt_disp_name)
            gt_disp = torch.Tensor(gt_disp.astype(np.float32)/256.).unsqueeze(0)

        input_imag = imag.to(device).unsqueeze(0)
        input_disp = disp.to(device).unsqueeze(0) / opt.max_disp
        
        if opt.use_cost:
            cost_name = os.path.join(cost_path, cost_names[idx])
            cost = np.load(cost_name)
            cost = torch.Tensor(cost)
            input_cost = cost.to(device).unsqueeze(dim=0)
        else:
            input_cost = None

        if opt.modal_type is not None:
            modal_name = os.path.join(modal_path,modal_names[idx])
            modal = io.imread(modal_name)
            modal = torch.Tensor(modal.astype(np.float32)/255.).unsqueeze(0)
            input_modal = modal.to(device).unsqueeze(0)
        else:
            input_modal = None

        pred = model(input_cost, input_disp, input_imag, input_modal)
        pred = pred.squeeze().cpu().detach().numpy()

        cv2.imwrite(os.path.join(opt.result_dir, image_names[idx][:-4] + ".png"), (pred * 255.0).astype(np.uint8))

        # AUC measurements
        disp = disp.squeeze().cpu().detach().numpy()
        gt_disp = gt_disp.squeeze().cpu().detach().numpy()
        valid = (gt_disp > 0) & (gt_disp < opt.max_disp)

        gt_disp = gt_disp[valid]
        disp = disp[valid]
        pred = pred[valid]
        pred = -pred

        ROC = []

        theta = 3
        intervals = 20

        quants = [100./intervals*t for t in range(1,intervals+1)]
        thresholds = [np.percentile(pred, q) for q in quants]
        subs = [pred <= t for t in thresholds]
        ROC_points = [(np.abs(disp - gt_disp) > theta)[s].mean() for s in subs]

        [ROC.append(r) for r in ROC_points]
        AUC = np.trapz(ROC, dx=1./intervals)
        AUCs.append(AUC)

        b3 = (np.abs(disp - gt_disp) > theta).mean()
        b3s.append(b3)
        opts.append(b3 + (1 - b3)*np.log(1 - b3))

        pbar.update(1)

pbar.close()

avg_AUC = np.mean(np.array(AUCs)) * 100.
avg_b3 = np.array(b3s).mean() * 100.
opt_AUC = np.array(opts).mean() * 100.

print("[ Dataset: %s | Stereo: %s | Opt. AUC: %.4f | Avg. AUC: %.4f | Avg. bad3: %.2f%% ]" % (opt.dataset_type, opt.stereo_type, opt_AUC, avg_AUC, avg_b3))
