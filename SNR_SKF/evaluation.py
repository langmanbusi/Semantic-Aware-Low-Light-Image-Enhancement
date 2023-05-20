import glob
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import cv2
import argparse
import utils

from natsort import natsort
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms.functional as TF

import lpips
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
        self.model = lpips.LPIPS(net=net)
        self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim, self.lpips]]

    def lpips(self, imgA, imgB, model=None):
        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        _, _, h, w = tA.shape
        # tA = TF.center_crop(tA, (h - (h % 16), w - (w % 16)))
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def SSIM(self, y_input, y_target):
        y_input = np.expand_dims(y_input, 0)
        y_target = np.expand_dims(y_target, 0)
        N, H, W, C = y_input.shape
        assert (C == 1 or C == 3)
        # N x C x H x W -> N x W x H x C -> N x H x W x C
        # y_input = np.swapaxes(y_input, 1, 3)
        # y_input = np.swapaxes(y_input, 1, 2)
        # y_target = np.swapaxes(y_target, 1, 3)
        # y_target = np.swapaxes(y_target, 1, 2)
        sum_structural_similarity_over_batch = 0.
        for i in range(N):
            if C == 3:
                sum_structural_similarity_over_batch += ssim(
                    y_input[i, :, :, :], y_target[i, :, :, :], multichannel=True)
            else:
                sum_structural_similarity_over_batch += ssim(
                    y_input[i, :, :, 0], y_target[i, :, :, 0])

        return sum_structural_similarity_over_batch / float(N)

    def ssim(self, imgA, imgB):
        # imgA = torch.Tensor(imgA).permute(2,0,1).unsqueeze(0)
        # imgB = torch.Tensor(imgB).permute(2,0,1).unsqueeze(0)
        # _, _, h, w = imgA.shape
        # score = utils.torchSSIM(imgA, imgB)

        # imgA = TF.center_crop(imgA, (h - (h % 16), w - (w % 16)))
        # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
        score, diff = ssim(imgA, imgB, full=True, multichannel=True)

        # score = utils.torchSSIM(torch.Tensor(imgA).permute(2,0,1).unsqueeze(0)/255, torch.Tensor(imgB).permute(2,0,1).unsqueeze(0)/255)
        return score

    def psnr(self, imgA, imgB):
        # imgA = torch.Tensor(imgA).permute(2,0,1).unsqueeze(0)
        # imgB = torch.Tensor(imgB).permute(2,0,1).unsqueeze(0)
        # _, _, h, w = imgA.shape
        # psnr_val = utils.torchPSNR(imgA, imgB)
        # imgA = TF.center_crop(imgA, (h - (h % 16), w - (w % 16)))

        psnr_val = psnr(imgA, imgB)

        # psnr_val = utils.torchPSNR(torch.Tensor(imgA).permute(2,0,1).unsqueeze(0)/255, torch.Tensor(imgB).permute(2,0,1).unsqueeze(0)/255)
        return psnr_val


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def format_result(psnr, ssim, lpips):
    return f'{psnr:0.3f}, {ssim:0.3f}, {lpips:0.3f}'

def measure_dirs(dirA, dirB, use_gpu, verbose=False):
    if verbose:
        vprint = lambda x: print(x)
    else:
        vprint = lambda x: None


    t_init = time.time()

    paths_A = fiFindByWildcard(os.path.join(dirA, f'*.{type}'))
    paths_B = fiFindByWildcard(os.path.join(dirB, f'*.{type}'))

    vprint("Comparing: ")
    vprint(dirA)
    vprint(dirB)

    measure = Measure(use_gpu=use_gpu)

    results = []
    for pathA, pathB in zip(paths_A, paths_B):
        result = OrderedDict()

        t = time.time()
        high = imread(pathA)
        low = imread(pathB)

        result['psnr'], result['ssim'], result['lpips'] = measure.measure(high, low)
        d = time.time() - t
        # vprint(f"{pathA.split('/')[-1]}, {pathB.split('/')[-1]}, {format_result(**result)}, {d:0.1f}")
        vprint(f"{pathA.split('/')[-1]}, {format_result(**result)}, {d:0.1f}")

        results.append(result)

    psnr = np.mean([result['psnr'] for result in results])
    ssim = np.mean([result['ssim'] for result in results])
    lpips = np.mean([result['lpips'] for result in results])

    vprint(f"Final Result: {format_result(psnr, ssim, lpips)}, {time.time() - t_init:0.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dirA', default='./results/LOLv2_model_cnn_sc_Seg_2.66_nclarge_validLQ_2/images/output/', type=str)
    parser.add_argument('-dirB', default='./results/LOLv2_model_cnn_sc_Seg_2.66_nclarge_validLQ_2/images/GT/', type=str)
    # parser.add_argument('-dirA', default='./results/LOLv2_real_model_Seg_2.66/images/output/', type=str)
    # parser.add_argument('-dirB', default='./results/LOLv2_real_model_Seg_2.66/images/GT/', type=str)

    # parser.add_argument('-dirA', default='./visual/abl-79-hwmnet-drbn/', type=str)
    # parser.add_argument('-dirB', default='./visual/high-79/', type=str)
    parser.add_argument('-type', default='png')
    parser.add_argument('--use_gpu', default=True)
    args = parser.parse_args()

    dirA = args.dirA
    dirB = args.dirB
    type = args.type
    use_gpu = args.use_gpu

    if len(dirA) > 0 and len(dirB) > 0:
        measure_dirs(dirA, dirB, use_gpu=use_gpu, verbose=True)
