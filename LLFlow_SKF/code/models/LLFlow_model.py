import logging
from collections import OrderedDict
from utils.util import get_resume_paths, opt_get

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from torch.cuda.amp import GradScaler, autocast
from models.modules.loss import hist_loss

logger = logging.getLogger('base')

class LayerSegNet(nn.Module):
    def __init__(self, channel=64, kernel_size=(3, 3)):
        super(LayerSegNet, self).__init__()

        self.relu = nn.LeakyReLU(0.1)
        self.conv1_1 = nn.Conv2d(3, channel, kernel_size=kernel_size, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=1, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.block1 = nn.Sequential(self.conv1_1, self.relu,
                                    self.conv1_2, self.relu, self.pool1)

        self.conv2_1 = nn.Conv2d(channel, 2 * channel, kernel_size=kernel_size, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(2 * channel, 2 * channel, kernel_size=kernel_size, stride=1, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.block2 = nn.Sequential(self.conv2_1, self.relu,
                                    self.conv2_2, self.relu, self.pool2)

        self.conv3_1 = nn.Conv2d(2 * channel, 4 * channel, kernel_size=kernel_size, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(4 * channel, 4 * channel, kernel_size=kernel_size, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(4 * channel, 4 * channel, kernel_size=kernel_size, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(4 * channel, 4 * channel, kernel_size=kernel_size, stride=1, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.block3 = nn.Sequential(self.conv3_1, self.relu,
                                    self.conv3_2, self.relu,
                                    self.conv3_3, self.relu,
                                    self.conv3_4, self.relu, self.pool3)

        self.conv4_1 = nn.Conv2d(4 * channel, 4 * channel, kernel_size=kernel_size, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(4 * channel, 4 * channel, kernel_size=kernel_size, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(4 * channel, 4 * channel, kernel_size=kernel_size, stride=1, padding=1)

        self.block4 = nn.Sequential(self.conv4_1, self.relu,
                                    self.conv4_2, self.relu,
                                    self.conv4_3, self.relu)

        self.deconv1 = nn.Conv2d(4 * channel, 2 * channel, kernel_size=kernel_size, stride=1, padding=1)
        self.deconv2 = nn.Conv2d(2 * channel, channel, kernel_size=kernel_size, stride=1, padding=1)
        self.deconv3 = nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=1, padding=1)

        self.conv_t1 = nn.Conv2d(4 * channel, channel, kernel_size=kernel_size, stride=1, padding=1)
        self.conv_t2 = nn.Conv2d(channel, channel, kernel_size=kernel_size, stride=1, padding=1)
        self.conv_t3 = nn.Conv2d(channel, 3, 1, stride=1, padding=0)

        self.block_t = nn.Sequential(self.conv_t1, self.relu, self.conv_t2, self.conv_t3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image):
        out1 = self.block1(image)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)

        fuse1 = F.interpolate(out3 + out4, [out2.shape[2], out2.shape[3]])
        out1_deconv = self.deconv1(fuse1)
        fuse2 = F.interpolate(out1_deconv + out2, [out1.shape[2], out1.shape[3]])
        out2_deconv = self.deconv2(fuse2)
        fuse3 = F.interpolate(out2_deconv + out1, [image.shape[2], image.shape[3]])
        out3_deconv = self.deconv3(fuse3)
        out1_deconv_upsample = F.interpolate(out1_deconv, [out3_deconv.shape[2], out3_deconv.shape[3]])
        out2_deconv_upsample = F.interpolate(out2_deconv, [out3_deconv.shape[2], out3_deconv.shape[3]])
        feature = torch.cat([out1_deconv_upsample, out2_deconv_upsample, out3_deconv], dim=1)
        out_feature = self.block_t(feature)
        pred = self.softmax(out_feature)

        return out_feature, pred, out1_deconv, out2_deconv, out3_deconv


class LLFlowModel(BaseModel):
    def __init__(self, opt, step):
        super(LLFlowModel, self).__init__(opt)
        self.opt = opt

        self.already_print_params_num = False

        self.heats = opt['val']['heats']
        self.n_sample = opt['val']['n_sample']
        self.hr_size = opt['datasets']['train']['GT_size']  # opt_get(opt, ['datasets', 'train', 'center_crop_hr_size'])
        # self.hr_size = 160 if self.hr_size is None else self.hr_size
        self.lr_size = self.hr_size // opt['scale']

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_Flow(opt, step).to(self.device)
        #
        weight_l1 = opt_get(self.opt, ['train', 'weight_l1']) or 0
        # if weight_l1 and 1:
        #     missing_keys, unexpected_keys = self.netG.load_state_dict(torch.load(
        #         '/home/yufei/project/LowLightFlow/experiments/to_pretrain_netG/models/1000_G.pth'),
        #         strict=False)
        #     print('missing %d keys, unexpected %d keys' % (len(missing_keys), len(unexpected_keys)))
        # if self.device.type != 'cpu':
        if opt['gpu_ids'] is not None and len(opt['gpu_ids']) > 0:
            if opt['dist']:
                self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            elif len(opt['gpu_ids']) > 1:
                self.netG = DataParallel(self.netG, opt['gpu_ids'])
            else:
                self.netG.cuda()
        # print network
        # self.print_network()

        if opt_get(opt, ['path', 'resume_state'], 1) is not None:
            self.load()
        else:
            print("WARNING: skipping initial loading, due to resume_state None")

        if self.is_train:
            self.netG.train()

            self.init_optimizer_and_scheduler(train_opt)
            self.log_dict = OrderedDict()

    def to(self, device):
        self.device = device
        self.netG.to(device)

    def init_optimizer_and_scheduler(self, train_opt):
        # optimizers
        self.optimizers = []
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        if isinstance(wd_G, str): wd_G = eval(wd_G)
        optim_params_RRDB = []
        optim_params_other = []
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            # print(k, v.requires_grad)
            if v.requires_grad:
                if '.RRDB.' in k:
                    optim_params_RRDB.append(v)
                    # print('opt', k)
                else:
                    optim_params_other.append(v)
                # if self.rank <= 0:
                #     logger.warning('Params [{:s}] will not optimize.'.format(k))

        print('rrdb params', len(optim_params_RRDB))

        self.optimizer_G = torch.optim.Adam(
            [
                {"params": optim_params_other, "lr": train_opt['lr_G'], 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': wd_G},
                {"params": optim_params_RRDB, "lr": train_opt.get('lr_RRDB', train_opt['lr_G']),
                 'beta1': train_opt['beta1'],
                 'beta2': train_opt['beta2'], 'weight_decay': 1e-5}
            ]
        )

        self.scaler = GradScaler()

        self.optimizers.append(self.optimizer_G)
        # schedulers
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                     restarts=train_opt['restarts'],
                                                     weights=train_opt['restart_weights'],
                                                     gamma=train_opt['lr_gamma'],
                                                     clear_state=train_opt['clear_state'],
                                                     lr_steps_invese=train_opt.get('lr_steps_inverse', [])))
        elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

    def add_optimizer_and_scheduler_RRDB(self, train_opt):
        # optimizers
        assert len(self.optimizers) == 1, self.optimizers
        assert len(self.optimizer_G.param_groups[1]['params']) == 0, self.optimizer_G.param_groups[1]
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                if '.RRDB.' in k:
                    self.optimizer_G.param_groups[1]['params'].append(v)
        assert len(self.optimizer_G.param_groups[1]['params']) > 0

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.real_H = data['GT'].to(self.device)  # GT

    def feed_data_1(self, LQ, GT, seg_map, seg_feature):
        self.var_L = LQ
        self.real_H = GT
        self.seg_map = seg_map
        self.seg_fea = seg_feature

    def get_module(self, model):
        if isinstance(model, nn.DataParallel):
            return model.module
        else:
            return model

    def optimize_color_encoder(self, step):
        self.netG.train()
        self.log_dict = OrderedDict()
        self.optimizer_G.zero_grad()
        color_lr, color_gt = self.fake_H[(0, 0)], logdet = self.netG(lr=self.var_L, gt=self.real_H,
                                                                     get_color_map=True)
        losses = {}
        total_loss = (color_gt - color_lr).abs().mean()
        # try:
        total_loss.backward()
        self.optimizer_G.step()
        mean = total_loss.item()
        return mean

    def optimize_parameters(self, step):
        train_RRDB_delay = opt_get(self.opt, ['network_G', 'train_RRDB_delay'])
        if train_RRDB_delay is not None and step > int(train_RRDB_delay * self.opt['train']['niter']) \
                and not self.get_module(self.netG).RRDB_training:
            if self.get_module(self.netG).set_rrdb_training(True):
                self.add_optimizer_and_scheduler_RRDB(self.opt['train'])

        # self.print_rrdb_state()

        self.netG.train()
        self.log_dict = OrderedDict()
        self.optimizer_G.zero_grad()
        # with autocast():
        losses = {}
        weight_fl = opt_get(self.opt, ['train', 'weight_fl'])
        weight_fl = 1 if weight_fl is None else weight_fl
        weight_l1 = opt_get(self.opt, ['train', 'weight_l1']) or 0
        flow_warm_up_iter = opt_get(self.opt, ['train', 'flow_warm_up_iter'])
        # print(step, flow_warm_up_iter)
        if flow_warm_up_iter is not None:
            if step > flow_warm_up_iter:
                weight_fl = 0
            else:
                weight_l1 = 0
        # print(weight_fl, weight_l1)
        if weight_fl > 0:
            if self.opt['optimize_all_z']:
                if self.opt['gpu_ids'] is not None and len(self.opt['gpu_ids']) > 0:
                    epses = [[] for _ in range(len(self.opt['gpu_ids']))]
                else:
                    epses = []
            else:
                epses = None
            z, nll, y_logits = self.netG(gt=self.real_H, lr=self.var_L, reverse=False, epses=epses,
                                         align_condition_feature=opt_get(self.opt,
                                                                         ['align_condition_feature']) or False,
                                         seg_map=self.seg_map, seg_fea=self.seg_fea)
            nll_loss = torch.mean(nll)
            losses['nll_loss'] = nll_loss * weight_fl

        if weight_l1 > 0:
            z = self.get_z(heat=0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
            sr, logdet = self.netG(lr=self.var_L, z=z, eps_std=0, reverse=True, reverse_with_grad=True)
            sr = sr.clamp(0, 1)
            not_nan_mask = ~torch.isnan(sr)
            sr[torch.isnan(sr)] = 0
            hl = 1000 * hist_loss(self.seg_map, sr, self.real_H)
            losses['hl_loss'] = hl
            # l1_loss = ((sr - self.real_H) * not_nan_mask).abs().mean()
            # losses['l1_loss'] = l1_loss * weight_l1
            # if flow_warm_up_iter is not None:
            #     print(l1_loss, not_nan_mask.float().mean())
        total_loss = sum(losses.values())
        # try:
        self.scaler.scale(total_loss).backward()
        if not self.already_print_params_num:
            logger.info("Parameters of full network %.4f and encoder %.4f"%(sum([m.numel() for m in self.netG.parameters() if m.grad is not None])/1e6, sum([m.numel() for m in self.netG.RRDB.parameters() if m.grad is not None])/1e6))
            self.already_print_params_num = True
        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        # except Exception as e:
        #     print(e)
        #     print(total_loss)

        mean = total_loss.item()
        return mean

    def print_rrdb_state(self):
        for name, param in self.get_module(self.netG).named_parameters():
            if "RRDB.conv_first.weight" in name:
                print(name, param.requires_grad, param.data.abs().sum())
        print('params', [len(p['params']) for p in self.optimizer_G.param_groups])

    def get_color_map(self):
        self.netG.eval()
        z = self.get_z(0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
        with torch.no_grad():
            color_lr, color_gt = self.fake_H[(0, 0)], logdet = self.netG(lr=self.var_L, gt=self.real_H,
                                                                         get_color_map=True)
        self.netG.train()
        return color_lr, color_gt

    def test(self):
        self.netG.eval()
        self.fake_H = {}
        if self.heats is not None:
            for heat in self.heats:
                for i in range(self.n_sample):
                    z = self.get_z(heat, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
                    with torch.no_grad():
                        self.fake_H[(heat, i)], logdet = self.netG(lr=self.var_L, z=z, eps_std=heat, reverse=True)
        else:
            z = self.get_z(0, seed=None, batch_size=self.var_L.shape[0], lr_shape=self.var_L.shape)
            with torch.no_grad():
                # torch.cuda.reset_peak_memory_stats()
                self.fake_H[(0, 0)], logdet = self.netG(lr=self.var_L, z=z.to(self.var_L.device), eps_std=0,
                                                        reverse=True, seg_map=self.seg_map, seg_fea=self.seg_fea)
                # from thop import clever_format, profile
                # print(clever_format(profile(self.netG, (None,self.var_L, z.to(self.var_L.device), 0 ,True))),"%.4")
                # print(torch.cuda.max_memory_allocated()/1024/1024/1024)
                # import time
                # t = time.time()
                # for i in range(15):
                #     with torch.no_grad():
                #         self.fake_H[(0, 0)], logdet = self.netG(lr=self.var_L, z=z.to(self.var_L.device), eps_std=0, reverse=True)
                # print((time.time()-t)/15)
        # with torch.no_grad():
        #     _, nll, _ = self.netG(gt=self.real_H, lr=self.var_L, reverse=False)
        self.netG.train()
        return None
        # return nll.mean().item()

    def get_encode_nll(self, lq, gt):
        self.netG.eval()
        with torch.no_grad():
            _, nll, _ = self.netG(gt=gt, lr=lq, reverse=False)
        self.netG.train()
        return nll.mean().item()

    def get_sr(self, lq, heat=None, seed=None, z=None, epses=None, seg_map=None, seg_ft=None):
        return self.get_sr_with_z(lq, heat, seed, z, epses, seg_map, seg_ft)[0]

    def get_encode_z(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, _, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise)
        self.netG.train()
        return z

    def get_encode_z_and_nll(self, lq, gt, epses=None, add_gt_noise=True):
        self.netG.eval()
        with torch.no_grad():
            z, nll, _ = self.netG(gt=gt, lr=lq, reverse=False, epses=epses, add_gt_noise=add_gt_noise)
        self.netG.train()
        return z, nll

    def get_sr_with_z(self, lq, heat=None, seed=None, z=None, epses=None, seg_map=None, seg_fea=None):
        self.netG.eval()
        if heat is None:
            heat = 0
        z = self.get_z(heat, seed, batch_size=lq.shape[0], lr_shape=lq.shape) if z is None and epses is None else z

        with torch.no_grad():
            sr, logdet = self.netG(lr=lq, z=z, eps_std=heat, reverse=True, epses=epses, seg_map=seg_map, seg_fea=seg_fea)
        self.netG.train()
        return sr, z

    def get_z(self, heat, seed=None, batch_size=1, lr_shape=None):
        if seed: torch.manual_seed(seed)
        if opt_get(self.opt, ['network_G', 'flow', 'split', 'enable']):
            C = self.get_module(self.netG).flowUpsamplerNet.C
            H = int(self.opt['scale'] * lr_shape[2] // self.get_module(self.netG).flowUpsamplerNet.scaleH)
            W = int(self.opt['scale'] * lr_shape[3] // self.get_module(self.netG).flowUpsamplerNet.scaleW)
            z = torch.normal(mean=0, std=heat, size=(batch_size, C, H, W)) if heat > 0 else torch.zeros(
                (batch_size, C, H, W))
        else:
            L = opt_get(self.opt, ['network_G', 'flow', 'L']) or 3
            fac = 2 ** L
            H = int(self.opt['scale'] * lr_shape[2] // self.get_module(self.netG).flowUpsamplerNet.scaleH)
            W = int(self.opt['scale'] * lr_shape[3] // self.get_module(self.netG).flowUpsamplerNet.scaleW)
            size = (batch_size, 3 * fac * fac, H, W)
            z = torch.normal(mean=0, std=heat, size=size) if heat > 0 else torch.zeros(size)
        return z

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        if self.heats is not None:
            for heat in self.heats:
                for i in range(self.n_sample):
                    out_dict[('NORMAL', heat, i)] = self.fake_H[(heat, i)].detach()[0].float().cpu()
        else:
            out_dict['NORMAL'] = self.fake_H[(0, 0)].detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        _, get_resume_model_path = get_resume_paths(self.opt)
        if get_resume_model_path is not None:
            self.load_network(get_resume_model_path, self.netG, strict=True, submodule=None)
            return

        load_path_G = self.opt['path']['pretrain_model_G']
        load_submodule = self.opt['path']['load_submodule'] if 'load_submodule' in self.opt['path'].keys() else 'RRDB'
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path'].get('strict_load', True),
                              submodule=None)

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
