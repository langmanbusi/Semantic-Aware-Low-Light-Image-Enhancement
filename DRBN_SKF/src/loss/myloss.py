import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from math import exp
from torch import nn



def gradient(input_tensor, direction):
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
    smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)

    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    grad_out = torch.abs(F.conv2d(input_tensor, kernel, stride=1, padding=1))
    return grad_out


def ave_gradient(input_tensor, direction):
    return F.avg_pool2d(gradient(input_tensor, direction), kernel_size=3, stride=1, padding=1)


def smooth(input1, input2):
    input2 = 0.299 * input2[:, 0, :, :] + 0.587 * input2[:, 1, :, :] + 0.114 * input2[:, 2, :, :]
    input2 = torch.unsqueeze(input2, dim=1)
    return torch.mean(gradient(input1, "x") * torch.exp(-10 * ave_gradient(input2, "x")) +
                      gradient(input1, "y") * torch.exp(-10 * ave_gradient(input2, "y")))


def l1loss(input1, input2):
    return F.l1_loss(input1, input2)


def grad_loss(input_I, input_R):
    N, C, H, W = input_I.size()
    if C == 3:
        input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        input_I = 0.299 * input_I[:, 0, :, :] + 0.587 * input_I[:, 1, :, :] + 0.114 * input_I[:, 2, :, :]
        input_I = torch.unsqueeze(input_I, dim=1)
    return torch.mean(gradient(input_I, "x") * torch.exp(-10 * ave_gradient(input_R, "x")) +
                      gradient(input_I, "y") * torch.exp(-10 * ave_gradient(input_R, "y")))


def grad_loss_1(input_I, input_R):
    return torch.mean(
        (gradient(input_I, "x") - gradient(input_R, "x")) ** 2 +
        (gradient(input_I, "y") - gradient(input_R, "y")) ** 2
    )


def mutual_consistency(I_low, I_high):
    low_gradient_x = gradient(I_low, "x")
    high_gradient_x = gradient(I_high, "x")
    M_gradient_x = low_gradient_x + high_gradient_x
    x_loss = M_gradient_x * torch.exp(-10 * M_gradient_x)
    low_gradient_y = gradient(I_low, "y")
    high_gradient_y = gradient(I_high, "y")
    M_gradient_y = low_gradient_y + high_gradient_y
    y_loss = M_gradient_y * torch.exp(-10 * M_gradient_y)
    mutual_loss = torch.mean(x_loss + y_loss)
    return mutual_loss


def illumination_smoothness(I, L):
    # L_transpose = L.permute(0, 2, 3, 1)
    # L_gray_transpose = 0.299*L[:,:,:,0] + 0.587*L[:,:,:,1] + 0.114*L[:,:,:,2]
    # L_gray = L.permute(0, 3, 1, 2)
    L_gray = 0.299 * L[:, 0, :, :] + 0.587 * L[:, 1, :, :] + 0.114 * L[:, 2, :, :]
    L_gray = L_gray.unsqueeze(dim=1)
    I_gradient_x = gradient(I, "x")
    L_gradient_x = gradient(L_gray, "x")
    epsilon = 0.01 * torch.ones_like(L_gradient_x)
    Denominator_x = torch.max(L_gradient_x, epsilon)
    x_loss = torch.abs(torch.div(I_gradient_x, Denominator_x))
    I_gradient_y = gradient(I, "y")
    L_gradient_y = gradient(L_gray, "y")
    Denominator_y = torch.max(L_gradient_y, epsilon)
    y_loss = torch.abs(torch.div(I_gradient_y, Denominator_y))
    mut_loss = torch.mean(x_loss + y_loss)
    return mut_loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM_loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma, gpu_id):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = self.centers.to(gpu_id)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        # y = x.sum()
        # x = x / (x.sum() + 0.0001)
        return x

    def forward_1(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
        x = x.sum(dim=-1)
        # x = x / (x.sum() + 0.00001)
        return x


def hist_loss(seg_pred, input_1, input_2, gpu_id):
    '''
    1. seg_pred transform to [1,2,3,2,3,1,3...] x batchsize
    2. Get class 1,2,3 index
    3. Use index to get value of img1 and img2
    4. Get hist of img1 and img2
    :return:
    '''
    N, C, H, W = seg_pred.shape
    bit = 256
    seg_pred = seg_pred.reshape(N, C, -1)
    seg_pred_cls = seg_pred.argmax(dim=1)
    input_1 = input_1.reshape(N, 3, -1)
    input_2 = input_2.reshape(N, 3, -1)
    # hist_1 = torch.zeros(N, 3 * C, bit).to(gpu_id)
    # hist_2 = torch.zeros(N, 3 * C, bit).to(gpu_id)
    soft_hist = SoftHistogram(bins=bit, min=0, max=1, sigma=400, gpu_id=gpu_id)
    loss = []
    # img:4,3,96,96  hist:4,9,256
    for n in range(N):
        # TODO 简化
        cls = seg_pred_cls[n]  # (H * W)
        img1 = input_1[n]
        img2 = input_2[n]
        for c in range(C):
            cls_index = torch.nonzero(cls == c).squeeze()
            img1_index = img1[:, cls_index]
            img2_index = img2[:, cls_index]
            for i in range(img1.shape[0]):
                img1_hist = soft_hist(img1_index[i])
                # h1 = torch.histc(img1_index[i], bins=bit, min=0, max=1)
                # h2 = torch.histc(img2_index[i], bins=bit, min=0, max=1)
                img2_hist = soft_hist(img2_index[i])
                loss.append(F.l1_loss(img1_hist, img2_hist))

    loss = sum(loss) / (N*C*H*W*3)
    return loss
