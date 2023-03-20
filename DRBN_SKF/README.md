
# DRBN-SKF
### [Paper](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_From_Fidelity_to_Perceptual_Quality_A_Semi-Supervised_Approach_for_Low-Light_CVPR_2020_paper.pdf) | [Project Page](https://github.com/flyywh/CVPR-2020-Semi-Low-Light) 
### Above are links to original paper and project, or you can just check the README_baseline.md for details of DRBN

## Get Started

Create environment

```
conda env create -f environment.yaml
```

### Pretrained Model
We provide the pre-trained models with the following settings:
- DRBN-SKF trained on LOL [[Google drive](https://drive.google.com/file/d/15djSbeDZd3NY5V-6XlRb-rk6_ljr-zLf/view?usp=sharing)] should be put in `./ckpt/LOL-DRBN-SKF/model_lol.pt`.
- DRBN-SKF trained on LOL-v2 [[Google drive](https://drive.google.com/file/d/1kO0Da29sCFF6vXwo7B_QuvrZvC31g0Ra/view?usp=sharing)] should be put in `./ckpt/LOLv2-DRBN-SKF/model_lol_v2.pt`.

Put the ckpts in the following way:

```
/ckpts
	/LOL-DRBN-SKF
		/model_lol.pt
	/LOLv2-DRBN-SKF
		/model_lol_v2.pt
```

### Test
You can check the training log to obtain the performance of the model. You can also directly test the performance of the pre-trained model as follows

1. Modify the paths to dataset and pre-trained model. You need to modify the following path in the `src/data/lowlighttest.py`.

```python
#### Test Settings
self.apath  = # path to test folder
self.dir_hr = os.path.join(self.apath, 'high') # make sure the folder name of normal-light images
self.dir_lr = os.path.join(self.apath, 'low')  # make sure the folder name of low-light images

#e.g., LOL and LOL-v2 dataset:
self.apath  = './LOL/test'
self.dir_hr = os.path.join(self.apath, 'high')
self.dir_lr = os.path.join(self.apath, 'low')
self.apath  = './LOLv2/Test'
self.dir_hr = os.path.join(self.apath, 'High')
self.dir_lr = os.path.join(self.apath, 'Low')
```

And you need to modify the following path in the `main_test.py`.

```python
#### Test Settings
my_model.model.load_state_dict(torch.load('./checkpoints/LOL-DRBN-SKF/model_lol.pt'))    # LOL dataset 
my_model.model.load_state_dict(torch.load('./checkpoints/LOL-DRBN-SKF/model_lol_v2.pt')) # LOL-v2 dataset
```

2. Test the model

To test the model with paired data and obtain the evaluation results, e.g., PSNR, SSIM, and LPIPS. You need to specify the data path and pre-trained model path mentioned above, and the results path in opt `--save` when running `main_test.py`. Then run `evaluation.py` to get metrics. Following are examples:

```bash
# generate results
python main_test.py --save_results --test_only --save your_result_dir 
# --save_results and --test_only are store_ture arguments. 
# evaluation
python evaluation.py --dirA ./LOL/test/high/ --dirB ./results/LOL/
```

You can check the output in `your_results_dir`.
### Train

1. Like the test settings, you should modify the paths to dataset in the `src/data/lowlight.py`, the code inside is similar to `src/data/lowlighttest.py`. You can specify the data path to choose the dataset (e.g., LOL and LOL-v2). We train the DRBN-SKF based on the original arguments of `./src/option.py`.

2. Train the network. 

Train on LOL and LOL-v2 dataset:

```bash
python train_LOL.py --save_results --save_models --save result_dir
# --save_results and --save_models are store_ture arguments. 
```

All logging files in the training process, e.g., log message, checkpoints, and snapshots, will be saved to `../experiments/results_dir/`, please change the name if you change settings.

## Contact
If you have any question, please feel free to contact us via xxxx.
