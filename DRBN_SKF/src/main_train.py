"""
Name    : main_train.py
Author  : xxxxxx
Time    : 2022/9/13 8:49
"""

import torch

import utility
import data
import model
import loss
from option import args
from trainer_seg_hist_fuse import Trainer



torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)

    my_model = model.Model(args, checkpoint)

    # my_model.model.load_state_dict(torch.load("/data1/xxxxxx/project/DRBL-stage1/"
    #                                           "train_hist_fuse_wo_pre_lol/model_124.pt"))


    args.n_colors = 3

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, my_model, loss, checkpoint, adv=True)
    while not t.terminate():
        t.train()
        t.test()





    checkpoint.done()
