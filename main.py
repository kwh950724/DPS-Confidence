import os
import argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PSMNet.models import *

import re
import chardet

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

parser = argparse.ArgumentParser(description="Modeling Stereo-Confidence Out of the End-to-End Stereo-Matching Network via Disparity Plane Sweep")
parser.add_argument("--dataset_type", type=str, default="kitti2012", choices=["kitti2012", "kitti2015", "mid2014"])
parser.add_argument("--dataset_dir", type=str, default="./samples/kitti-stereo-2012", help="dataset directory path")
parser.add_argument("--stereo_type", type=str, default="psmnet")
parser.add_argument("--pretrained_path", type=str, default="./PSMNet/pretrained/pretrained_model_KITTI2012.tar", help="pre-trained weights path of stereo-matching network")
parser.add_argument("--result_dir", type=str, default="./samples", help="result directory path")

parser.add_argument("--max_disp", type=int, default=192, help="maximum disparity value of stereo-matching network")
parser.add_argument("--num_shifts", type=int, default=5, help="number of disparity plane shifts (N)")
parser.add_argument("--max_shift", type=int, default=2, help="maximum disparity plane shift (K)")
parser.add_argument("--sigma", type=float, default=133.084, help="confidence scale factor")
parser.add_argument("--dps", type=int, nargs="+", default=None)

opt = parser.parse_args()

opt.result_disp_dir = os.path.join(opt.dataset_dir, "disp_0")
os.makedirs(opt.result_disp_dir, exist_ok=True)
opt.result_disp_dir = os.path.join(opt.result_disp_dir, opt.stereo_type)
os.makedirs(opt.result_disp_dir, exist_ok=True)

opt.result_cost_dir = os.path.join(opt.dataset_dir, "cost")
os.makedirs(opt.result_cost_dir, exist_ok=True)
opt.result_cost_dir = os.path.join(opt.result_cost_dir, opt.stereo_type)
os.makedirs(opt.result_cost_dir, exist_ok=True)

opt.result_conf_dir = os.path.join(opt.dataset_dir, "conf")
os.makedirs(opt.result_conf_dir, exist_ok=True)
opt.result_conf_dir = os.path.join(opt.result_conf_dir, opt.stereo_type)
os.makedirs(opt.result_conf_dir, exist_ok=True)

normalize = {"mean": [0.485, 0.456, 0.406],
              "std": [0.229, 0.224, 0.225]}
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(**normalize)])

if opt.dataset_type == "kitti2012":
    img_l_dir = os.path.join(opt.dataset_dir, "colored_0")
    img_r_dir = os.path.join(opt.dataset_dir, "colored_1")
    disp_dir = os.path.join(opt.dataset_dir, "disp_occ")

    filenames = sorted(os.listdir(os.path.join(img_l_dir)))
    filenames = [x for x in filenames if x.find("_10.png") != -1]
elif opt.dataset_type == "kitti2015":
    img_l_dir = os.path.join(opt.dataset_dir, "image_2")
    img_r_dir = os.path.join(opt.dataset_dir, "image_3")
    disp_dir = os.path.join(opt.dataset_dir, "disp_occ_0")

    filenames = sorted(os.listdir(os.path.join(img_l_dir)))
    filenames = [x for x in filenames if x.find("_10.png") != -1]
elif opt.dataset_type == "mid2014":
    img_l_dir = os.path.join(opt.dataset_dir, "im0")
    img_r_dir = os.path.join(opt.dataset_dir, "im1")
    disp_dir = os.path.join(opt.dataset_dir, "disp0GT")

    filenames = sorted(os.listdir(os.path.join(img_l_dir)))
    filenames = [x for x in filenames if x.find(".png") != -1]

# Compute disparity plane shifts (k_{i}) based on the number of disparity plane shifts (N) and the maximum disparity plane shift (K) values.
if opt.dps is None:
    opt.dps = [opt.max_shift - shift for shift in range(opt.num_shifts)]
else:
    opt.num_shifts = len(opt.dps)
    opt.max_shift = max(opt.dps)
opt.dps.sort()

opt.idx_zero = None
for i in range(len(opt.dps)):
    if opt.dps[i] == 0:
        opt.idx_zero = i
if opt.idx_zero is None:
    print("Error: Zero-Shift (0) Must Be Included in Disparity Plane Shifts for obtaining Ideal (Desirable) Disparity Profile.")
    print("Current Disparity Plane Shifts:")
    print(opt.dps)
    exit()

# Target stereo-matching network
if opt.stereo_type == "psmnet":
    model = stackhourglass(opt.max_disp)

model = nn.DataParallel(model).cuda()

if opt.pretrained_path is not None:
    state_dict = torch.load(opt.pretrained_path)
    model.load_state_dict(state_dict["state_dict"])

model.eval()

print("========================================")
print("Dataset Info: [ Type: %s | Directory: %s ]" % (opt.dataset_type, opt.dataset_dir))
print("Result Info: [ Directory: %s ]" % (opt.result_dir))
print("Stereo Info: [ Type: %s | Pre-trained Weights Path: %s ]" % (opt.stereo_type, opt.pretrained_path))
print("Model Info: [ Maximum Disparity: %d | Number of Shifts: %d | Maximum Disparity Shift: %d | Sigma: %.3f ]" % (opt.max_disp, opt.num_shifts, opt.max_shift, opt.sigma))
print("Disparity Plane Shifts:")
print(opt.dps)
print("========================================")

with torch.no_grad():
    aucs = []
    opts = []
    b3s = []

    pbar = tqdm(total=len(filenames))

    for idx, filename in enumerate(filenames):
        if opt.dataset_type == "kitti2015" or opt.dataset_type == "kitti2012":
            img_l = Image.open(os.path.join(img_l_dir, filename)).convert("RGB")
            img_r = Image.open(os.path.join(img_r_dir, filename)).convert("RGB")
            disp_gt = cv2.imread(os.path.join(disp_dir, filename), cv2.IMREAD_ANYDEPTH) / 256.0
            valid = (disp_gt > 0.0) & (disp_gt < opt.max_disp) 
        elif opt.dataset_type == "mid2014":
            img_l = Image.open(os.path.join(img_l_dir, filename)).convert("RGB")
            img_r = Image.open(os.path.join(img_r_dir, filename)).convert("RGB")
            disp_gt, _ = readPFM(os.path.join(disp_dir, filename[:-4] + ".pfm"))
            disp_gt[disp_gt == np.inf] = 0
            valid = (disp_gt > 0.0) & (disp_gt < opt.max_disp) 

        img_l = transform(img_l).cuda()
        img_r = transform(img_r).cuda()

        C, H, W = img_l.shape

        if img_l.shape[1] % 16 != 0:
            times = img_l.shape[1] // 16
            pad_top = (times + 1) * 16 - img_l.shape[1]
        else:
            pad_top = 0

        if img_l.shape[2] % 16 != 0:
            times = img_l.shape[2] // 16
            pad_right = (times + 1) * 16 - img_l.shape[2]
        else:
            pad_right = 0

        # ==================== #
        # 1. Obtain I_L^{DPS} and I_R^{DPS} from input stereo image pair using disparity plane sweep.
        img_l = F.pad(img_l, (0, pad_right, pad_top, 0)).unsqueeze(dim=0)
        # I_L^{DPS}
        img_l_dps = img_l.repeat(opt.num_shifts, 1, 1, 1)

        # I_R^{DPS}
        img_r_dps = None
        for idx_shift, shift in enumerate(opt.dps):
            if shift > 0:
                img_r_temp = img_r[:,:,shift:]
                img_r_temp = torch.cat([img_r_temp, torch.zeros(C, H, abs(shift)).cuda()], dim=-1) 
            elif shift < 0:
                img_r_temp = img_r[:,:,:shift]
                img_r_temp = torch.cat([torch.zeros(C, H, abs(shift)).cuda(), img_r_temp], dim=-1) 
            elif shift == 0:
                img_r_temp = img_r

            img_r_temp = F.pad(img_r_temp, (0, pad_right, pad_top, 0)).unsqueeze(dim=0)

            if idx_shift == 0:
                img_r_dps = img_r_temp
            else:
                img_r_dps = torch.cat([img_r_dps, img_r_temp], dim=0)
        # ==================== #
        # 2. Obtain D_s^{DPS} and D_tgt^{DPS} by forwarding I_L^{DPS} and I_R^{DPS} through the target stereo-matching network.
        disp_pred_dps, cost = model(img_l_dps, img_r_dps)
        disp_pred_dps = disp_pred_dps.data.squeeze(dim=1).cpu().numpy()
        cost = torch.topk(cost[opt.idx_zero], k=7, dim=0)[0].cpu().numpy()

        if pad_top != 0 and pad_right == 0:
            disp_pred_dps = disp_pred_dps[:,pad_top:,:]
            cost = cost[:,pad_top:,:]
        elif pad_top == 0 and pad_right != 0:
            disp_pred_dps = disp_pred_dps[:,:,:-pad_right]
            cost = cost[:,:,:-pad_right]
        elif pad_top != 0 and pad_right != 0:
            disp_pred_dps = disp_pred_dps[:,pad_top:,:-pad_right]
            cost = cost[:,pad_top:,:-pad_right]

        # Zero-shifted disparity map D_pred^{0}.
        disp_pred_0 = disp_pred_dps[opt.idx_zero]
        cv2.imwrite(os.path.join(opt.result_disp_dir, filename[:-4] + ".png"), (disp_pred_0 * 256.0).astype(np.uint16))

        # Save top-k matching cost
        np.save(os.path.join(opt.result_cost_dir, filename[:-4] + ".npy"), cost)

        # Obtain the ideal (desirable) disparity profile using zero-shifted disparity map and disparity plane shifts.
        disp_tgt_dps = np.expand_dims(disp_pred_dps[opt.idx_zero].copy(), axis=0)
        disp_tgt_dps = np.repeat(disp_tgt_dps, repeats=opt.num_shifts, axis=0) 
        for i in range(opt.num_shifts):
            disp_tgt_dps[i,:,:] = disp_tgt_dps[i,:,:] + opt.dps[i]

        disp_pred_dps = np.delete(disp_pred_dps, opt.idx_zero, axis=0)
        disp_tgt_dps = np.delete(disp_tgt_dps, opt.idx_zero, axis=0)
        # ==================== #
        # 3. Compute unreliability (U) and obtain confidence map (C). 
        # Mask out disparity values outside of the range due to the disparity plane shift.
        mask_inlier = (disp_tgt_dps >= 0) & (disp_tgt_dps <= opt.max_disp)
        mask_outlier = (disp_tgt_dps < 0) | (disp_tgt_dps > opt.max_disp)
        denom = np.sum(mask_inlier, axis=0)
        unreliablity = np.abs(disp_pred_dps - disp_tgt_dps)
        unreliablity[mask_outlier] = 0
        unreliablity = np.sum(unreliablity, axis=0) / denom

        unreliablity_norm = unreliablity / opt.max_disp
        confidence = np.exp(-opt.sigma * unreliablity_norm)

        cv2.imwrite(os.path.join(opt.result_conf_dir, filename[:-4] + ".png"), (confidence * 255.0).astype(np.uint8))
        # ==================== #

        if (opt.dataset_type == "kitti2012") and (idx < 20):
            pbar.update(1)
            continue

        confidence_gt = np.abs(disp_pred_0 - disp_gt) <= 3
        disp_gt = disp_gt[valid]
        disp_pred_0 = disp_pred_0[valid]
        confidence = confidence[valid]
        confidence = -confidence

        roc = []

        theta = 3
        intervals = 20

        quants = [100./intervals*t for t in range(1,intervals+1)]
        thresholds = [np.percentile(confidence, q) for q in quants]
        subs = [confidence <= t for t in thresholds]
        roc_points = [(np.abs(disp_pred_0 - disp_gt) > theta)[s] for s in subs]
        roc_points_ = []
        for roc_point in roc_points:
            if len(roc_point) == 0:
                roc_points_.append(0)
            else:
                roc_point = roc_point.mean()
                roc_points_.append(roc_point)

        [roc.append(r) for r in roc_points_]
        auc = np.trapz(roc, dx=1./intervals)
        aucs.append(auc)

        b3 = (np.abs(disp_pred_0 - disp_gt) > theta).mean()
        b3s.append(b3)
        opts.append(b3 + (1 - b3)*np.log(1 - b3))

        pbar.update(1)

    pbar.close()

    opt_auc = np.array(opts).mean() * 100.
    avg_auc = np.mean(np.array(aucs)) * 100.
    avg_b3 = np.array(b3s).mean() * 100.
    
    print("========================================")
    print("[ Dataset: %s | Stereo: %s | Opt. AUC: %.4f | Avg. AUC: %.4f | Avg. bad3: %.2f%% ]\n" % (opt.dataset_type, opt.stereo_type, opt_auc, avg_auc, avg_b3))
