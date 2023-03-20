
# LLFlow-SKF
### [Paper](https://arxiv.org/pdf/2109.05923.pdf) | [Project Page](https://wyf0912.github.io/LLFlow/) 
### Above are links to original paper and project, or you can just check the README_baseline.md for details of LLFlow

## Get Started

Create environment

```
conda env create -f environment.yaml
```

### Pre-trained Model
We provide the pre-trained models with the following settings:
- LLFlow-S-SKF trained on LOL [[Google drive](https://drive.google.com/file/d/16PihEBd0GxbU4L0ImbI86X-Gk_KVSXU1/view?usp=share_link)] with training config file `./confs/LOL-S-SKF.yml`
- LLFlow-S-SKF trained on LOL-v2 [[Google drive](https://drive.google.com/file/d/1KefXtxKK9HheG5o7-Y2IkrncD6UEffVk/view?usp=share_link)] with training config file `./confs/LOLv2-S-SKF.yml`.
- LLFlow-L-SKF trained on LOL [[Google drive](https://drive.google.com/file/d/1ItJcHGsegkwkWcUvlFmE6WYAKDZGkAcC/view?usp=share_link)] with training config file `./confs/LOL-L-SKF.yml`
- LLFlow-L-SKF trained on LOL-v2 [[Google drive](https://drive.google.com/file/d/11TNkFDUgI7P0NX2bpDGlhx9ip0cvQJSa/view?usp=share_link)] with training config file `./confs/LOLv2-L-SKF.yml`.

Put the ckpts in the following way:

```
/ckpts
	/LOL-LLFlow-SKF
		/LOL_LLFlow_S_SKF.pth
		/LOL_LLFlow_L_SKF.pth
	/LOLv2-LLflow-SKF
		/LOLv2_LLFlow_S_SKF.pth
		/LOLv2_LLFlow_L_SKF.pth
```

Please download pre-trained model [[Onedrive](https://1drv.ms/u/s!Aus8VCZ_C_33f5Bfbt4KmLeX8uw)] from the github of [[HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/master)], and put the ckpt in the following way:

```
LLFlow_SKF
	/hrseg
		/hrnet_w48_pascal_context_cls59_480x480.pth
```

### Test
You can check the training log to obtain the performance of the model. You can also directly test the performance of the pre-trained model as follows

1. Modify the paths to dataset and pre-trained model. You need to modify the following path in the config files in `./confs`

```python
#### Test Settings
dataroot_GT # needed for testing with paired data
dataroot_LR # needed for testing with paired data
model_path
```

2. Test the model

To test the model with paired data and obtain the evaluation results, e.g., PSNR, SSIM, and LPIPS. You need to specify the data path ```dataroot_LR```, ```dataroot_GT```, and pre-trained model path ```model_path``` in the config file. Then run

```bash
# generate results
python test.py --opt your_config_path
# You need to specify an appropriate config file since it stores the config of the model, e.g., the number of layers.
# evaluation
python evaluation.py --dirA ./LOL/test/high/ --dirB ./results/LOL/
```

You can check the output in `../results`.
### Train

1. Modify the paths to dataset in the config yaml files. We provide the following training configs for both `LOL` and `LOL-v2` benchmarks. You can also create your own configs for your own dataset.

```bash
.\confs\LOL-S-SKF.yml
.\confs\LOL-L-SKF.yml
.\confs\LOLv2-S-SKF.yml
.\confs\LOLv2-S-SKF.yml
```
You need to modify the following terms 

```python
datasets.train.root
datasets.val.root
gpu_ids: [0] 
# Our model can be trained using a single GPU with memory>20GB. You can also train the model using multiple GPUs by adding more GPU ids in it.
# LLFlow-L-SKF need GPU memory>30GB, LLFlow-S-SKF need GPU memory>20GB.
path.pretrain_model_G # if you want to train with a init pretrain model
```
2. Train the network.

```bash
python train.py --opt your_config_path
```

All logging files in the training process, e.g., log message, checkpoints, and snapshots, will be saved to `./experiments/conf['name']/`, please change the name if you change settings.

## Contact
If you have any question, please feel free to contact us via xxxx.
