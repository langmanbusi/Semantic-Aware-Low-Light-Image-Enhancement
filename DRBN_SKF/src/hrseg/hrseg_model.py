import torch
import torch.nn as nn
from hrseg.hrseg_lib.models import seg_hrnet
from hrseg.hrseg_lib.config import config
from hrseg.hrseg_lib.config import update_config
import argparse
import os
from glob import glob
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from hrseg.hrseg_lib.datasets.cityscapes import Cityscapes
from hrseg.hrseg_lib.datasets.pascal_ctx import PASCALContext
from hrseg.hrseg_lib.utils.modelsummary import get_model_summary
import logging
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# console = logging.StreamHandler()
# logging.getLogger('').addHandler(console)
def create_hrnet():
    args = {}
    args['cfg'] = './hrseg/hrseg_lib/pascal_ctx/seg_hrnet_w48_cls59_480x480_sgd_lr4e-3_wd1e-4_bs_16_epoch200.yaml'
    args['opt'] = []
    update_config(config, args)
    if torch.__version__.startswith('1'):
        module = eval('seg_hrnet')
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval(config.MODEL.NAME + '.get_seg_model')(config)
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print('Seg parameters: {}'.format(params))
    # print(model)
    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    # if config.TEST.MODEL_FILE:
    #     model_state_file = config.TEST.MODEL_FILE
    # else:
    #     model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    # logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load('./hrseg/hrnet_w48_pascal_context_cls59_480x480.pth')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #     logger.info(
    #         '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('HRNet load')
    return model
# model = model.cuda()
def padtensor(input_):
    mul = 16
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    return input_

if __name__ == '__main__':
    model = create_hrnet().cuda()
    model.eval()
    test_low_data_names = glob('./LOL/' + 'test/low/' + '*.*')
    test_low_data_names.sort()
    test_high_data_names = glob('./LOL/' + 'test/high/' + '*.*')
    test_high_data_names.sort()
    dataset = Cityscapes()
    # dataset = PASCALContext()
    N = len(test_low_data_names)
    with torch.no_grad():
        for idx in tqdm(range(N)):
            test_low_img_path = test_low_data_names[idx]
            test_high_img_path = test_high_data_names[idx]
            # show name of result image
            test_img_name = test_low_img_path.split('/')[-1]
            # change dim
            test_low_img = Image.open(test_low_img_path)
            test_high_img = Image.open(test_high_img_path)

            test_low_img = np.array(test_low_img, dtype="float32") / 255.0
            test_low_img = np.transpose(test_low_img, (2, 0, 1))
            test_high_img = np.array(test_high_img, dtype="float32")
            test_high_img = np.transpose(test_high_img, (2, 0, 1)) / 255.0

            _, h, w = test_low_img.shape
            input_low_test = torch.from_numpy(np.expand_dims(test_low_img, axis=0)).cuda()
            input_high_test = torch.from_numpy(np.expand_dims(test_high_img, axis=0)).cuda()
            # input_low_test = padtensor(input_low_test)
            # input_high_test = padtensor(input_high_test)

            low_out = model(input_low_test)

            high_out = model(input_high_test)

            low_out = F.interpolate(low_out[0], [h, w], mode='bilinear', align_corners=False)
            high_out = F.interpolate(high_out[0], [h, w], mode='bilinear', align_corners=False)

            filepath = './results/LOL_Seg'

            if not os.path.exists(filepath):
                os.makedirs(filepath)
            dataset.save_pred(low_out, filepath, name=test_img_name[:-4] + '_low')
            dataset.save_pred(high_out, filepath, name=test_img_name[:-4] + '_high')
            # low = torch.clamp(low_out, 0, 1)
            # low = low[:,:,:h,:w]
            # high = torch.clamp(high_out, 0, 1)
            # high = high[:,:,:h,:w]

            # low_image = np.squeeze(low_out.cpu().numpy())
            # high_image = np.squeeze(high_out.cpu().numpy())
            #
            # low_im = Image.fromarray(np.clip(low_image * 255.0, 0, 255.0).astype('uint8'))
            # high_im = Image.fromarray(np.clip(high_image * 255.0, 0, 255.0).astype('uint8'))

