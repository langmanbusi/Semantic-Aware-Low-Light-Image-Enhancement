# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn

from model.Fuseblock import TransformerBlock


def make_model(args, parent=False):
    return DRBN(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        feat1 = self.convs(x)
        feat2 = self.LFF(feat1) + x
        return feat2

class DRBN(nn.Module):
    def __init__(self, args):
        super(DRBN, self).__init__()

        self.recur1 = DRBN_BU(args)
        self.recur2 = DRBN_BU(args)
        self.recur3 = DRBN_BU(args)
        self.recur4 = DRBN_BU(args, True)
        # self.recur4 = DRBN_BU(args, False)

    def forward(self, x_input):
        x = x_input
        
        res_g1_s1, res_g1_s2, res_g1_s4, feat_g1_s1, feat_g1_s2, feat_g1_s4 = self.recur1([0, torch.cat((x, x), 1), 0, 0, 0, 0, 0, 0])
        res_g2_s1, res_g2_s2, res_g2_s4, feat_g2_s1, feat_g2_s2, feat_g2_s4 = self.recur2([1, torch.cat((res_g1_s1, x), 1), res_g1_s1, res_g1_s2, res_g1_s4, feat_g1_s1, feat_g1_s2, feat_g1_s4])
        res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4 = self.recur3([1, torch.cat((res_g2_s1, x), 1), res_g2_s1, res_g2_s2, res_g2_s4, feat_g2_s1, feat_g2_s2, feat_g2_s4])
        res_g4_s1, res_g4_s2, res_g4_s4, feat_g4_s1, feat_g4_s2, feat_g4_s4 = self.recur4([1, torch.cat((res_g3_s1, x), 1), res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4])

        return res_g4_s1, res_g4_s2, res_g4_s4

    def forward_1(self, x_input):
        x = x_input

        res_g1_s1, res_g1_s2, res_g1_s4, feat_g1_s1, feat_g1_s2, feat_g1_s4 = self.recur1([0, torch.cat((x, x), 1), 0, 0, 0, 0, 0, 0])
        res_g2_s1, res_g2_s2, res_g2_s4, feat_g2_s1, feat_g2_s2, feat_g2_s4 = self.recur2([1, torch.cat((res_g1_s1, x), 1), res_g1_s1, res_g1_s2, res_g1_s4, feat_g1_s1, feat_g1_s2, feat_g1_s4])
        res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4 = self.recur3([1, torch.cat((res_g2_s1, x), 1), res_g2_s1, res_g2_s2, res_g2_s4, feat_g2_s1, feat_g2_s2, feat_g2_s4])

        return res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4

    def forward_2(self, x, res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4, seg_orin, seg_fea):

        res_g4_s1, res_g4_s2, res_g4_s4, feat_g4_s1, feat_g4_s2, feat_g4_s4 = self.recur4([1, torch.cat((res_g3_s1, x), 1), res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4], seg_orin, seg_fea)

        return res_g4_s1, res_g4_s2, res_g4_s4


class DRBN_BU(nn.Module):
    def __init__(self, args, seg=False):
        super(DRBN_BU, self).__init__()

        G0 = 16
        kSize = 3
        self.D = 6 
        G = 8
        C = 4

        self.SFENet1 = nn.Conv2d(args.n_colors*2, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        self.RDBs = nn.ModuleList()
        
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = 2*G0, growRate = 2*G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = 2*G0, growRate = 2*G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )
        self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.UPNet2 = nn.Sequential(*[
                nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.UPNet4 = nn.Sequential(*[
                nn.Conv2d(G0*2, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, 3, kSize, padding=(kSize-1)//2, stride=1)
            ])

        self.Down1 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=2)
        self.Down2 = nn.Conv2d(G0, G0*2, kSize, padding=(kSize-1)//2, stride=2)

        self.Up1 = nn.ConvTranspose2d(G0, G0, kSize+1, stride=2, padding=1)
        self.Up2 = nn.ConvTranspose2d(G0*2, G0, kSize+1, stride=2, padding=1) 

        self.Relu = nn.ReLU()
        self.Img_up = nn.Upsample(scale_factor=2, mode='bilinear')

        if seg:
            self.fuseblk_1 = TransformerBlock(96, 32)
            self.fuseblk_2 = TransformerBlock(48, 16)
            self.fuseblk_3 = TransformerBlock(59, 16)

    def part_forward(self, x,seg_orin, seg_fea):
        #
        # Stage 1
        #
        flag = x[0]
        input_x = x[1]

        prev_s1 = x[2]
        prev_s2 = x[3]
        prev_s4 = x[4]

        prev_feat_s1 = x[5]
        prev_feat_s2 = x[6]
        prev_feat_s4 = x[7]

        f_first = self.Relu(self.SFENet1(input_x))
        f_s1  = self.Relu(self.SFENet2(f_first))
        f_s2 = self.Down1(self.RDBs[0](f_s1)) 
        f_s4 = self.Down2(self.RDBs[1](f_s2))
       
        if flag == 0: 
            f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4))
            f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4))
            f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2))+f_first
        elif seg_orin is not None:
            f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4)) + prev_feat_s4  # （bs, 32, 64)
            f_s4 = self.fuseblk_1(f_s4, seg_fea[1])
            f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4)) + prev_feat_s2  # （bs, 16, 128)
            f_s2 = self.fuseblk_2(f_s2, seg_fea[0])
            f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2)) + f_first + prev_feat_s1  # （bs, 16, 256)
            f_s1 = self.fuseblk_3(f_s1, seg_orin)
        else:
            f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4)) + prev_feat_s4      # （bs, 32, 64)
            f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4)) + prev_feat_s2          # （bs, 16, 128)
            f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2))+f_first + prev_feat_s1  # （bs, 16, 256)

        res4 = self.UPNet4(f_s4)
        res2 = self.UPNet2(f_s2) + self.Img_up(res4)
        res1 = self.UPNet(f_s1) + self.Img_up(res2)

        return res1, res2, res4, f_s1, f_s2, f_s4


    def forward(self, x_input,seg_orin=None, seg_fea=None):
        x = x_input

        res1, res2, res4, f_s1, f_s2, f_s4 = self.part_forward(x, seg_orin, seg_fea)

        return res1, res2, res4, f_s1, f_s2, f_s4
