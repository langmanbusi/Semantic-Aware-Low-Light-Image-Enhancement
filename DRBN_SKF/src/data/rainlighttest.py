import os
from data import srdata

class RainLightTest(srdata.SRData):
    def __init__(self, args, name='RainLightTest', train=True, benchmark=False):
        super(RainLightTest, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(RainLightTest, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(RainLightTest, self)._set_filesystem(dir_data)
        self.apath = '/home/yangwenhan/pytorch_project/data_dir/rain_removal/rain_light_test/'
        print(self.apath)
        self.dir_hr = os.path.join(self.apath, 'norain')
        self.dir_lr = os.path.join(self.apath, 'rain')

