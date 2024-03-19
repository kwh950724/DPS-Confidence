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
import time
from os import listdir
from skimage import io
import models
import datasets
from lossfunc import BCE2d
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="LAFNet (CVPR 2019) stereo confidence training")
parser.add_argument("--seed", type=int, default=1, metavar="S")
parser.add_argument("--log_dir", type=str, default="./logs")
parser.add_argument("--kitti2012_dir", type=str, default="../samples/kitti-stereo-2012", help="kitti2012 dataset directory")
parser.add_argument("--kitti2015_dir", type=str, default="../samples/kitti-stereo-2015", help="kitti2015 dataset directory")
parser.add_argument("--mid2014_dir", type=str, default="../samples/middlebury-2014", help="mid2014 dataset directory")
parser.add_argument("--base_lr", type=float, default=3e-3, help="base_lr")
parser.add_argument("--batch_size", type=float, default=4, help="batch_size")
parser.add_argument("--start_epoch", type=int, default=0, help="starting epoch")
parser.add_argument("--num_epochs", type=int, default=2000, help="num_epochs")
parser.add_argument("--step_size_lr", type=int, default=1000, help="step_size_lr")
parser.add_argument("--gamma_lr", type=float, default=0.1, help="gamma_lr")
parser.add_argument("--stereo_type", type=str, default="psmnet", choices=["psmnet", "ganet", "raft", "igev", "leastereo", "acvnet", "sttr"])
parser.add_argument("--max_disp", type=float, default=192.0, help="maximum disparity of stereo-matching network")
parser.add_argument("--modal_type", type=str, default=None, help="additional input modality type")
parser.add_argument("--use_cost", action="store_true", default=False)
parser.add_argument("--weights_path", type=str, default=None)
parser.add_argument("--eval_frequency", type=int, default=100)
opt = parser.parse_args()

torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(opt.log_dir):
    os.mkdir(opt.log_dir)

opt.log_dir = os.path.join(opt.log_dir, opt.stereo_type)
if not os.path.exists(opt.log_dir):
    os.mkdir(opt.log_dir)

if opt.modal_type is not None:
    opt.log_dir = os.path.join(opt.log_dir, opt.modal_type)
else:
    opt.log_dir = os.path.join(opt.log_dir, "none") 
if not os.path.exists(opt.log_dir):
    os.mkdir(opt.log_dir)

if opt.use_cost:
    opt.log_dir = os.path.join(opt.log_dir, "use_cost")
else:
    opt.log_dir = os.path.join(opt.log_dir, "no_cost")
if not os.path.exists(opt.log_dir):
    os.mkdir(opt.log_dir)

opt.param_path = os.path.join(opt.log_dir, "params.txt")
opt.metric_path = os.path.join(opt.log_dir, "metrics.txt")

if not os.path.exists(opt.param_path):
    f = open(opt.param_path, "w")
else:
    os.remove(opt.param_path)
    f = open(opt.param_path, "w")

f.write("[ Seed: %d | Stereo Type: %s | Max Disparity: %d | Modal Type: %s | Cost: %s ]" % (opt.seed, opt.stereo_type, int(opt.max_disp), opt.modal_type, str(opt.use_cost)))
f.close()

def train():
    model = models.LAFNet_CVPR2019(modal_type=opt.modal_type,
                                   use_cost=opt.use_cost).to(device)

    if opt.weights_path is not None:
        model.load_state_dict(torch.load(opt.weights_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size_lr, gamma=opt.gamma_lr)

    train_data = datasets.KITTI2012_train(dataset_dir=opt.kitti2012_dir,
                                          stereo_type=opt.stereo_type,
                                          max_disp=opt.max_disp,
                                          modal_type=opt.modal_type,
                                          use_cost=opt.use_cost)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=True)

    test_data = datasets.KITTI2015_test(dataset_dir=opt.kitti2015_dir,
                                        stereo_type=opt.stereo_type,
                                        max_disp=opt.max_disp,
                                        modal_type=opt.modal_type,
                                        use_cost=opt.use_cost)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=1,
                                              shuffle=False)

    test_data_kitti2012 = datasets.KITTI2012_test(dataset_dir=opt.kitti2012_dir,
                                                  stereo_type=opt.stereo_type,
                                                  max_disp=opt.max_disp,
                                                  modal_type=opt.modal_type,
                                                  use_cost=opt.use_cost)
    test_loader_kitti2012 = torch.utils.data.DataLoader(dataset=test_data_kitti2012,
                                                        batch_size=1,
                                                        shuffle=False)                                  

    test_data_mid2014 = datasets.MID2014_test(dataset_dir=opt.mid2014_dir,
                                                     stereo_type=opt.stereo_type,
                                                     max_disp=opt.max_disp,
                                                     modal_type=opt.modal_type,
                                                     use_cost=opt.use_cost)
    test_loader_mid2014 = torch.utils.data.DataLoader(dataset=test_data_mid2014,
                                                      batch_size=1,
                                                      shuffle=False)   

    prev_loss = 1.0
    kitti2012_best_auc = 100.0
    kitti2012_best_epoch = 0
    kitti2015_best_auc = 100.0
    kitti2015_best_epoch = 0
    mid2014_best_auc = 100.0
    mid2014_best_epoch = 0
    
    for epoch in range(opt.start_epoch, opt.start_epoch + opt.num_epochs):
        model.train()
        count = 0
        train_loss = 0.
        
        for batch_idx, data in enumerate(train_loader):
            disp = data["disp"].to(device)
            imag = data["imag"].to(device)
            gt = data["gt"].to(device)
            cost = None
            modal = None
            if opt.use_cost:
                cost = data["cost"].to(device)
            if opt.modal_type is not None:
                modal = data["modal"].to(device)

            optimizer.zero_grad()

            output = model(cost, disp, imag, modal)

            loss = BCE2d(output, gt)

            loss.backward()
            optimizer.step()

            train_loss += loss
            count += 1

        train_loss = train_loss / count
        
        if train_loss <= prev_loss:
            print("Epoch: {}/{} [ Avg. Loss: {:.4f} ]".format(epoch + 1, opt.num_epochs, train_loss))
            prev_loss = train_loss
        
        scheduler.step()

        if (epoch + 1) % opt.eval_frequency == 0:
            torch.save(model.state_dict(), os.path.join(opt.log_dir, str(epoch + 1) + ".pth"))

            model.eval()

            with torch.no_grad():
                AUCs = []
                opts = []
                b3s = []

                for batch_idx, inputs in enumerate(test_loader):
                    imag = inputs["imag"].cuda()
                    disp = inputs["disp"].cuda() / opt.max_disp
                    disp_pred = inputs["disp"].cuda()
                    disp_gt = inputs["gt_disp"].cuda()
                    cost = None
                    modal = None
                    if opt.use_cost:
                        cost = inputs["cost"].cuda()
                    if opt.modal_type is not None:
                        modal = inputs["modal"].cuda()

                    pred = model(cost, disp, imag, modal)

                    pred = pred.squeeze().cpu().detach().numpy()
                    
                    disp_pred = disp_pred.squeeze().cpu().detach().numpy()
                    disp_gt = disp_gt.squeeze().cpu().detach().numpy()

                    valid = (disp_gt > 0) & (disp_gt < opt.max_disp)
                    disp_gt = disp_gt[valid]
                    disp_pred = disp_pred[valid]
                    pred = pred[valid]
                    pred = -pred

                    ROC = []

                    theta = 3
                    intervals = 20

                    quants = [100./intervals*t for t in range(1,intervals+1)]
                    thresholds = [np.percentile(pred, q) for q in quants]
                    subs = [pred <= t for t in thresholds]
                    ROC_points = [(np.abs(disp_pred - disp_gt) > theta)[s].mean() for s in subs]

                    [ROC.append(r) for r in ROC_points]
                    AUC = np.trapz(ROC, dx=1./intervals)
                    AUCs.append(AUC)

                    b3 = (np.abs(disp_pred - disp_gt) > theta).mean()
                    b3s.append(b3)
                    opts.append(b3 + (1 - b3)*np.log(1 - b3))

                opt_AUC = np.array(opts).mean() * 100.
                avg_AUC = np.mean(np.array(AUCs)) * 100.
                avg_b3 = np.array(b3s).mean() * 100.

                print("=========================================================")
                print("KITTI 2015 Test Done. Epoch: %d/%d [ Opt. AUC: %.4f | Avg. AUC: %.4f | Avg. bad3: %.2f%% ]" % (epoch + 1, opt.num_epochs, opt_AUC, avg_AUC, avg_b3))

                metric = "Epoch: {:d}/{:d} [ Opt. AUC: {:.4f} | Avg. AUC: {:.4f} | Avg. bad3: {:.2f}% ]\n".format(epoch + 1, opt.num_epochs, opt_AUC, avg_AUC, avg_b3)

                if not os.path.exists(opt.metric_path):
                    f = open(opt.metric_path, "w")
                else:
                    f = open(opt.metric_path, "a")
                f.write(metric)
                f.close()

                if kitti2015_best_auc >= avg_AUC:
                    kitti2015_best_auc = avg_AUC
                    kitti2015_best_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(opt.log_dir, "kitti2015.pth"))

                AUCs = []
                opts = []
                b3s = []

                for batch_idx, inputs in enumerate(test_loader_kitti2012):
                    imag = inputs["imag"].cuda()
                    disp = inputs["disp"].cuda() / opt.max_disp
                    disp_pred = inputs["disp"].cuda()
                    disp_gt = inputs["gt_disp"].cuda()
                    cost = None
                    modal = None
                    if opt.use_cost:
                        cost = inputs["cost"].cuda()
                    if opt.modal_type is not None:
                        modal = inputs["modal"].cuda()

                    pred = model(cost, disp, imag, modal)

                    pred = pred.squeeze().cpu().detach().numpy()
                    
                    disp_pred = disp_pred.squeeze().cpu().detach().numpy()
                    disp_gt = disp_gt.squeeze().cpu().detach().numpy()

                    valid = (disp_gt > 0) & (disp_gt < opt.max_disp)
                    disp_gt = disp_gt[valid]
                    disp_pred = disp_pred[valid]
                    pred = pred[valid]
                    pred = -pred

                    ROC = []

                    theta = 3
                    intervals = 20

                    quants = [100./intervals*t for t in range(1,intervals+1)]
                    thresholds = [np.percentile(pred, q) for q in quants]
                    subs = [pred <= t for t in thresholds]
                    ROC_points = [(np.abs(disp_pred - disp_gt) > theta)[s].mean() for s in subs]

                    [ROC.append(r) for r in ROC_points]
                    AUC = np.trapz(ROC, dx=1./intervals)
                    AUCs.append(AUC)

                    b3 = (np.abs(disp_pred - disp_gt) > theta).mean()
                    b3s.append(b3)
                    opts.append(b3 + (1 - b3)*np.log(1 - b3))

                opt_AUC = np.array(opts).mean() * 100.
                avg_AUC = np.mean(np.array(AUCs)) * 100.
                avg_b3 = np.array(b3s).mean() * 100.

                print("KITTI 2012 Test Done. Epoch: %d/%d [ Opt. AUC: %.4f | Avg. AUC: %.4f | Avg. bad3: %.2f%% ]" % (epoch + 1, opt.num_epochs, opt_AUC, avg_AUC, avg_b3))

                metric = "Epoch: {:d}/{:d} [ Opt. AUC: {:.4f} | Avg. AUC: {:.4f} | Avg. bad3: {:.2f}% ]\n".format(epoch + 1, opt.num_epochs, opt_AUC, avg_AUC, avg_b3)

                if not os.path.exists(opt.metric_path):
                    f = open(opt.metric_path, "w")
                else:
                    f = open(opt.metric_path, "a")
                f.write(metric)

                f.close()

                if kitti2012_best_auc >= avg_AUC:
                    kitti2012_best_auc = avg_AUC
                    kitti2012_best_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(opt.log_dir, "kitti2012.pth"))

                AUCs = []
                opts = []
                b3s = []

                for batch_idx, inputs in enumerate(test_loader_mid2014):
                    imag = inputs["imag"].cuda()
                    disp = inputs["disp"].cuda() / opt.max_disp
                    disp_pred = inputs["disp"].cuda()
                    disp_gt = inputs["gt_disp"].cuda()
                    mask = inputs["mask"]
                    cost = None
                    modal = None
                    if opt.use_cost:
                        cost = inputs["cost"].cuda()
                    if opt.modal_type is not None:
                        modal = inputs["modal"].cuda()

                    pred = model(cost, disp, imag, modal)

                    pred = pred.squeeze().cpu().detach().numpy()
                    
                    disp_pred = disp_pred.squeeze().cpu().detach().numpy()
                    disp_gt = disp_gt.squeeze().cpu().detach().numpy()
                    mask = mask.squeeze().cpu().detach().numpy()

                    valid = (disp_gt > 0) & (disp_gt < opt.max_disp)
                    disp_gt = disp_gt[valid]
                    disp_pred = disp_pred[valid]
                    pred = pred[valid]
                    pred = -pred

                    ROC = []

                    theta = 3
                    intervals = 20

                    quants = [100./intervals*t for t in range(1,intervals+1)]
                    thresholds = [np.percentile(pred, q) for q in quants]
                    subs = [pred <= t for t in thresholds]
                    ROC_points = [(np.abs(disp_pred - disp_gt) > theta)[s].mean() for s in subs]

                    [ROC.append(r) for r in ROC_points]
                    AUC = np.trapz(ROC, dx=1./intervals)
                    AUCs.append(AUC)

                    b3 = (np.abs(disp_pred - disp_gt) > theta).mean()
                    b3s.append(b3)
                    opts.append(b3 + (1 - b3)*np.log(1 - b3))

                opt_AUC = np.array(opts).mean() * 100.
                avg_AUC = np.mean(np.array(AUCs)) * 100.
                avg_b3 = np.array(b3s).mean() * 100.

                print("MID 2014 Test Done. Epoch: %d/%d [ Opt. AUC: %.4f | Avg. AUC: %.4f | Avg. bad3: %.2f%% ]" % (epoch + 1, opt.num_epochs, opt_AUC, avg_AUC, avg_b3))
                print("=========================================================")

                metric = "Epoch: {:d}/{:d} [ Opt. AUC: {:.4f} | Avg. AUC: {:.4f} | Avg. bad3: {:.2f}% ]\n".format(epoch + 1, opt.num_epochs, opt_AUC, avg_AUC, avg_b3)

                if not os.path.exists(opt.metric_path):
                    f = open(opt.metric_path, "w")
                else:
                    f = open(opt.metric_path, "a")
                f.write(metric)
                f.write("=========================================================\n")

                f.close()

                if mid2014_best_auc >= avg_AUC:
                    mid2014_best_auc = avg_AUC
                    mid2014_best_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(opt.log_dir, "mid2014.pth"))

    summary = "Summary: [ KITTI 2015 Top AUC: {:.4f} at Epoch {:d} | KITTI 2012 Top AUC: {:.4f} at Epoch {:d} | MID 2014 Top AUC: {:.4f} at Epoch {:d} ]\n".format(kitti2015_best_auc, kitti2015_best_epoch, kitti2012_best_auc, kitti2012_best_epoch, mid2014_best_auc, mid2014_best_epoch)

    if not os.path.exists(opt.metric_path):
        f = open(opt.metric_path, "w")
    else:
        f = open(opt.metric_path, "a")
    f.write(summary)

    f.close()
    
if __name__ == "__main__":
    train()
