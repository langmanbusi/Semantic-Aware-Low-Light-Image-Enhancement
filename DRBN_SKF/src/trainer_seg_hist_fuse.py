"""
Name    : trainer_seg_hist_fuse.py
Author  : xxxxxx
Time    : 2022/9/12 21:08
"""

import os
from decimal import Decimal

import utility
import torch.nn as nn

import torch
from tqdm import tqdm
import pytorch_ssim
import torchvision
from PIL import Image

from hrseg.hrseg_model import create_hrnet
from loss.myloss import hist_loss

import torch.optim as optim

from model.D_Net import Discriminator as D_Net
from model.D_Net import calculate_loss_D, calculate_loss_G





def tensor_save_rgbimage(tensor, filename, cuda=False):
    if cuda:
        img = tensor.clone().cpu().clamp(0, 255).numpy()
    else:
        img = tensor.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


class vgg_v2(nn.Module):
    def __init__(self, vgg_model):
        super(vgg_v2, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }

    def forward(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output


def vgg_loss(vgg, img, gt):
    mse = nn.MSELoss(size_average=True)
    img_vgg = vgg(img)
    gt_vgg = vgg(gt)

    # return 0.4*mse(img_vgg[2], gt_vgg[2]) + 0.2*mse(img_vgg[3], gt_vgg[3])
    return mse(img_vgg[0], gt_vgg[0]) + 0.6 * mse(img_vgg[1], gt_vgg[1]) + 0.4 * mse(img_vgg[2], gt_vgg[2]) + 0.2 * mse(
        img_vgg[3], gt_vgg[3])


def vgg_init(vgg_loc):
    vgg_model = torchvision.models.vgg16(pretrained=False).cuda()
    vgg_model.load_state_dict(torch.load(vgg_loc))
    trainable(vgg_model, False)

    return vgg_model


def trainable(net, trainable):
    for para in net.parameters():
        para.requires_grad = trainable


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp, adv=False):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = my_model
        self.loss = my_loss

        "adv"
        self.adv =adv

        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()

        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        criterion_ssim = pytorch_ssim.SSIM(window_size=11)
        criterion_mse = nn.MSELoss(size_average=True)

        #    vgg_model = vgg_init('./pretrained/vgg16-397923af.pth')
        #    vgg = vgg_v2(vgg_model)
        #    vgg.eval()

        'define seg model'
        seg_model = create_hrnet().cuda()
        seg_model.eval()

        "define discriminator and its optimizer"
        if self.adv:
            model_D = D_Net().cuda()
            optimizer_D = optim.Adam(model_D.parameters(), lr=float(lr/1000), betas=(0.9, 0.999), eps=1e-8)

        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            lr = lr / 255.0
            hr = hr / 255.0

            [b, c, h, w] = hr.shape

            # phr1, phr2, phr4 = self.model(lr, 3)
            res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4 = self.model.forward_1(lr, 3)

            'use seg_model'
            seg_map, seg_orin, seg_fea = seg_model(res_g3_s1)

            phr1, phr2, phr4 = self.model.forward_2(lr, res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2,
                                                    feat_g3_s4,seg_orin, seg_fea)

            hr4 = hr[:, :, 0::4, 0::4]
            hr2 = hr[:, :, 0::2, 0::2]
            hr1 = hr

            'use seg_model'
            seg_map, seg_orin, seg_fea = seg_model(phr1)

            if self.adv:
                loss_D = calculate_loss_D(model_D, hr1, phr1, seg_map)

                optimizer_D.zero_grad()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

                hist_loss_ = hist_loss(seg_map, phr1, hr1, gpu_id='cuda')
                rect_loss = criterion_ssim(phr1, hr1) + criterion_ssim(phr2, hr2) + criterion_ssim(phr4, hr4)
                loss_G = calculate_loss_G(model_D, hr1, phr1, seg_map)

                full_loss = rect_loss + hist_loss_ + 0.1 * loss_G
                self.optimizer.zero_grad()
                full_loss.backward()
                self.optimizer.step()


            else:
                'use hist loss'
                hist_loss_ = hist_loss(seg_map, phr1, hr1, gpu_id='cuda')

                rect_loss = criterion_ssim(phr1, hr1) + criterion_ssim(phr2, hr2) + criterion_ssim(phr4, hr4)

                full_loss = rect_loss + hist_loss_

                if full_loss.item() < self.args.skip_threshold * self.error_last:
                    full_loss.backward()
                    self.optimizer.step()
                else:
                    print('Skip this batch {}! (Loss: {})'.format(
                        batch + 1, rect_loss.item()
                    ))

            timer_model.hold()


            if (batch + 1) % self.args.print_every == 0:
                if self.adv:
                    self.ckp.write_log('[{}/{}]\t{}\t{}\t{}\tD: {}\tG: {}\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        full_loss.item(),
                        rect_loss.item(),
                        hist_loss_.item(),
                        loss_D.item(),
                        loss_G.item(),
                        # percept_loss.item(),
                        timer_model.release(),
                        timer_data.release()))
                else:
                    self.ckp.write_log('[{}/{}]\t{}\t{}\t{}\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        full_loss.item(),
                        rect_loss.item(),
                        hist_loss_.item(),
                        # percept_loss.item(),
                        timer_model.release(),
                        timer_data.release()))

            timer_data.tic()

        # print(rect_loss.item())

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        'define seg model'
        seg_model = create_hrnet().cuda()
        seg_model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)

                for idx_img, (lr, hr, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    lr = lr / 255.0
                    hr = hr / 255.0

                    [b, c, h, w] = hr.shape
                    n_map = torch.zeros(b, c, h, w).cuda()

                    # phr1, phr2, phr4 = self.model(lr, 3)
                    res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4 = self.model.forward_1(lr, 3)

                    'use seg_model'
                    seg_map, seg_orin, seg_fea = seg_model(res_g3_s1)

                    phr1, phr2, phr4 = self.model.forward_2(lr, res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2,
                                                            feat_g3_s4, seg_orin, seg_fea)


                    phr = utility.quantize(phr1 * 255, self.args.rgb_range)
                    lr = utility.quantize(lr * 255, self.args.rgb_range)
                    hr = utility.quantize(hr * 255, self.args.rgb_range)

                    save_list = [hr, lr, phr, lr]

                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            phr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale, epoch)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs


