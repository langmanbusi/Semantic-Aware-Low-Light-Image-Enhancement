# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn

def make_model(args, parent=False):
    return Wave_rdn(args)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = True

    def forward(self, x):
        return iwt_init(x)

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2

    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


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
        #G0 = growRate0
        #G  = growRate
        #C  = nConvLayers
        G0 = 16
        G  = 16
        C  = 4

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class Wave_rdn(nn.Module):
    def __init__(self, args):
        super(Wave_rdn, self).__init__()

        G0 = 16
        kSize = 3
        self.D = 2
        G = 16
        C = 4

        self.DWT = DWT()
        self.IWT = IWT()

        # Shallow feature extraction net
        self.SFENet1_direct = nn.Conv2d(3, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2_direct = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=2)

        self.SFENet3_direct = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet4_direct = nn.Conv2d(G0, 16*3, kSize, padding=(kSize-1)//2, stride=2)

        # Redidual dense blocks and dense feature fusion
        self.RDBs_1 = nn.ModuleList()
        for i in range(self.D):
            self.RDBs_1.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF_1 = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Redidual dense blocks and dense feature fusion
        self.RDBs_2 = nn.ModuleList()
        for i in range(self.D):
            self.RDBs_2.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF_2 = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G0, 256*3, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(256*3, 256*3, kSize, padding=(kSize-1)//2, stride=1, groups=16),
                nn.Conv2d(256*3, 16*3, kSize, padding=(kSize-1)//2, stride=1, groups=16)
            ]) 

    def forward(self, x):
        x_ori = x

        x_l1 = self.DWT(x)
        x_l2 = self.DWT(x_l1)

        x = self.SFENet1_direct(x)
        RDBs_out1 = []
        for i in range(self.D):
            x = self.RDBs_1[i](x)
            RDBs_out1.append(x)

        x_res1 = self.SFENet2_direct(self.GFF_1(torch.cat(RDBs_out1,1)))

        x = self.SFENet3_direct(x_res1)
        RDBs_out2 = []
        for i in range(self.D):
            x = self.RDBs_2[i](x)
            RDBs_out2.append(x)

        x_res2 = self.SFENet4_direct(self.GFF_2(torch.cat(RDBs_out2,1)))

        y_l2 = x_res2 #self.UPNet(x) 
        y_l1 = self.IWT(y_l2)
        y = self.IWT(y_l1)
        return y, x_l2, y_l2
