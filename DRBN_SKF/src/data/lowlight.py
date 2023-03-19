import os
from data import srdata
import glob

class LowLight(srdata.SRData):
    def __init__(self, args, name='LowLight', train=True, benchmark=False):
        super(LowLight, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        super(LowLight, self)._set_filesystem(dir_data)
        self.apath = './LOL/test/'
        print(self.apath)
        self.dir_hr   = os.path.join(self.apath, 'high')
        self.dir_lr   = os.path.join(self.apath, 'low')

        # self.apath = './LOLv2/Test/'
        # print(self.apath)
        # self.dir_hr = os.path.join(self.apath, 'Our_normal')
        # self.dir_lr = os.path.join(self.apath, 'Our_low')

    def _scan(self):
        names_hr, names_lr = super(LowLight, self)._scan()
        names_hr   = names_hr[self.begin - 1:self.end]
        names_lr   = names_lr[self.begin - 1:self.end]

        return names_hr, names_lr
