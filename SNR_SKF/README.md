
# SNR-SKF
### [Paper](https://jiaya.me/publication/) | [Project Page](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance) 
### Above are links to original paper and project, or you can just check the README_baseline.md for details of LLFlow

## Get Started
Create environment

```
conda env create -f environment.yaml
```
### Pre-trained Model
We provide the pre-trained models with the following settings:
- SNR-SKF trained on LOL [[Google drive](https://drive.google.com/file/d/1QbYqIgg2fmu0Y2fyWg2uAMLLJ_PqIfa0/view?usp=share_link)] with training config file `./options/train/LOLv1-SKF.yml`
- SNR-SKF trained on LOL-v2 [[Google drive](https://drive.google.com/file/d/1VVQu0pMAMiJIhKXknau-LIiByKArXs4o/view?usp=sharing)] with training config file `./options/train/LOLv2-SKF.yml`.

Put the ckpts in the following way:

```
/pretrain_model
	/SNR_SKF_LOL.pth
	/SNR_SKF_LOLv2.pth

```

Please download pre-trained model [[Onedrive](https://1drv.ms/u/s!Aus8VCZ_C_33f5Bfbt4KmLeX8uw)] from the github of [[HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/master)], and put the ckpt in the following way:

```
SNR_SKF
	/model
		/hrseg_lib
			/hrnet_w48_pascal_context_cls59_480x480.pth
```

### Test
You can check the training log to obtain the performance of the model. You can also directly test the performance of the pre-trained model as follows

1. Modify the paths to dataset and pre-trained model. You need to modify the following path in the config files in `./options/test/`

```python
datasets:
  	test:
		dataroot_GT # needed for testing with paired data
		dataroot_LR # needed for testing with paired data

path:
	pretrain_model_G:
```

2. Test the model

To test the model with paired data and obtain the evaluation results, e.g., PSNR, SSIM, and LPIPS. You need to specify the data path ```dataroot_LR```, ```dataroot_GT```, and pre-trained model path ```pretrain_model_G ``` in the config file. Then run

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
./options/train/LOLv1_SKF.yml
./options/train/LOLv2_SKF.yml
```
You need to modify the following terms 

```python
datasets.train.dataroot_GT/LQ
datasets.val.dataroot_GT/LQ
gpu_ids: [0] 
# SNR-SKF need GPU memory about 21GB.
path.pretrain_model_G # if you want to train with a init pretrain model
```
2. Train the network.

```bash
python train.py --opt your_config_path
```

All logging files in the training process, e.g., log message, checkpoints, and snapshots, will be saved to `./experiments/conf['name']/`, please change the name if you change settings.

## Contact
If you have any question, please feel free to contact us via xxxx.
