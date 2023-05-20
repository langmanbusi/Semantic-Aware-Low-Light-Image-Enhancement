import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss, CharbonnierLoss2, hist_loss
from thop import profile

import time

logger = logging.getLogger('base')


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        # if opt['dist']:
        #     self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        # else:
        #     self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'cb2':
                self.cri_pix = CharbonnierLoss2().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']    # 1.0

            self.cri_pix_ill = nn.L1Loss(reduction='sum').to(self.device)

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        self.nf = data['nf'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)
        self.seg_map = None
        self.seg_fea = None

    def feed_data_1(self, LQ, nf, GT, seg_map, seg_feature):
        self.var_L = LQ
        self.nf = nf
        self.real_H = GT
        self.seg_map = seg_map
        self.seg_fea = seg_feature


    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        dark = self.var_L
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        light = self.nf
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)

        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max+0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()

        self.fake_H = self.netG(self.var_L, mask, seg_map=self.seg_map, seg_fea=self.seg_fea)

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)

        if self.seg_map is not None and step % 50 == 0:
            l_hist = hist_loss(self.seg_map, self.fake_H, self.real_H)
            l_final = l_pix + 10000 * l_hist
        else:
            l_final = l_pix

        l_final.backward()
        self.optimizer_G.step()
        self.log_dict['l_pix'] = l_pix.item()
        if self.seg_map is not None and step % 50 == 0:
            self.log_dict['l_hist'] = l_hist.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max+0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()
            # flops, params = profile(self.netG, inputs=(self.var_L, mask, self.seg_map, self.seg_fea))
            # print(flops / 1e9, params / 1e6)  # TODO Rebuttal

            self.fake_H = self.netG(self.var_L, mask, seg_map=self.seg_map, seg_fea=self.seg_fea)

        self.netG.train()

    def test4_seg(self):
        self.netG.eval()
        self.fake_H = None
        with torch.no_grad():
            B, C, H, W = self.var_L.size()

            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max + 0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()

            del light
            del dark
            del noise
            torch.cuda.empty_cache()

            var_L = self.var_L.clone().view(B, C, H, W)
            H_new = 400
            W_new = 608
            var_L = F.interpolate(var_L, size=[H_new, W_new], mode='bilinear')
            mask = F.interpolate(mask, size=[H_new, W_new], mode='bilinear')
            var_L = var_L.view(B, C, H_new, W_new)
            self.fake_H = self.netG(var_L, mask, seg_map=self.seg_map, seg_fea=self.seg_fea)
            self.fake_H = F.interpolate(self.fake_H, size=[H, W], mode='bilinear')

            del var_L
            del mask
            torch.cuda.empty_cache()

        self.netG.train()

        return self.fake_H

    def test4(self):
        self.netG.eval()
        self.fake_H = None
        with torch.no_grad():
            B, C, H, W = self.var_L.size()

            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max + 0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()

            del light
            del dark
            del noise
            torch.cuda.empty_cache()

            var_L = self.var_L.clone().view(B, C, H, W)
            # H_new = 400
            # W_new = 608
            var_L = self.padtensor(var_L)
            mask = self.padtensor(mask)
            # var_L = F.interpolate(var_L, size=[H_new, W_new], mode='bilinear')
            # mask = F.interpolate(mask, size=[H_new, W_new], mode='bilinear')
            # var_L = var_L.view(B, C, H_new, W_new)
            self.fake_H = self.netG(var_L, mask, seg_map=self.seg_map, seg_fea=self.seg_fea)
            self.fake_H = self.fake_H[:,:,:H,:W]
            # self.fake_H = F.interpolate(self.fake_H, size=[H, W], mode='bilinear')

            del var_L
            del mask
            torch.cuda.empty_cache()

        self.netG.train()


    def test5(self):
        self.netG.eval()
        self.fake_H = None
        with torch.no_grad():
            B, C, H, W = self.var_L.size()

            dark = self.var_L
            dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
            if not (len(self.nf.shape) == 4):
                self.nf = self.nf.unsqueeze(dim=0)
            light = self.nf
            light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
            noise = torch.abs(dark - light)
            mask = torch.div(light, noise + 0.0001)

            batch_size = mask.shape[0]
            height = mask.shape[2]
            width = mask.shape[3]
            mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
            mask_max = mask_max.view(batch_size, 1, 1, 1)
            mask_max = mask_max.repeat(1, 1, height, width)
            mask = mask * 1.0 / (mask_max + 0.0001)

            mask = torch.clamp(mask, min=0, max=1.0)
            mask = mask.float()

            del light
            del dark
            del noise
            torch.cuda.empty_cache()

            var_L = self.var_L.clone().view(B, C, H, W)
            H_new = 384
            W_new = 384
            var_L = F.interpolate(var_L, size=[H_new, W_new], mode='bilinear')
            mask = F.interpolate(mask, size=[H_new, W_new], mode='bilinear')
            var_L = var_L.view(B, C, H_new, W_new)
            self.fake_H = self.netG(var_L, mask)
            self.fake_H = F.interpolate(self.fake_H, size=[H, W], mode='bilinear')

            del var_L
            del mask
            torch.cuda.empty_cache()

        self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()

        dark = self.var_L
        dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
        if not(len(self.nf.shape) == 4):
            self.nf = self.nf.unsqueeze(dim=0)
        light = self.nf
        light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
        noise = torch.abs(dark - light)
        mask = torch.div(light, noise + 0.0001)

        batch_size = mask.shape[0]
        height = mask.shape[2]
        width = mask.shape[3]
        mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
        mask_max = mask_max.view(batch_size, 1, 1, 1)
        mask_max = mask_max.repeat(1, 1, height, width)
        mask = mask * 1.0 / (mask_max + 0.0001)

        mask = torch.clamp(mask, min=0, max=1.0)
        mask = mask.float()
        mask = mask.repeat(1, 3, 1, 1)
        out_dict['rlt3'] = mask[0].float().cpu()

        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LQ2'] = self.fake_H
        out_dict['ill'] = mask[0].float().cpu()
        out_dict['rlt2'] = self.nf.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()

        del dark
        del light
        del mask
        del noise
        del self.real_H
        del self.nf
        del self.var_L
        del self.fake_H
        torch.cuda.empty_cache()
        return out_dict


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            # logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def padtensor(self, input_):
        mul = 16
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
        padh = H - h if h % mul != 0 else 0
        padw = W - w if w % mul != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        return input_

