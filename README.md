# Inbed_pose_estimation: Multimodal In-bed Pose and Shape Estimation under the Blankets

### Installation
- Clone this repo:
``` 
git clone https://github.com/AnonymousSubmission43/Inbed_pose_estimation
cd Inbed_pose_estimation
```
- Dependencies:
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/).
All dependencies for defining the environment are provided in `environment/py3pt110.yaml`.


### Preparing your Data
- Download the data from [here](https://drive.google.com/file/d/11gnOHa0VXYmLVR_fCH0c4_zR4zaCxRS2/view?usp=sharing)
- Refer to `config.py` to define the data paths:
```
DATA_ROOT = 'dir/to/Data_pose/' 
```

### Pretrained Models
Please download the pretrained model from [here](https://drive.google.com/drive/folders/1f-ZdVmjUPdPJGkYJkJJMFkyEEG8-RVmG?usp=sharing).

Please download other dependencies from [here](https://drive.google.com/file/d/10fpg5-NaDPjxzmTnyfSHGW13AtVoq_W4/view?usp=sharing), and put unzip the zip file and put it under 'Inbed_pose_estimation/'. So the unzip folder would have the structure:
```
Inbed_pose_estimation/cdf37_0-dist 
Inbed_pose_estimation/data
Inbed_pose_estimation/pyopengl
```

### Train
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --name train_slp_4mod --model cashmrV2  --run_smplify \
	--data_train slp-4mod-train --data_test slp-4mod-uncover+slp-4mod-cover1+slp-4mod-cover2 \
	--no_render --batch_size 32 --num_cas_iters 3
```


### Test
```
python eval.py --model cashmrV2 --checkpoint ../checkpoints/epoch_85_0.pt --result_file ../test
```
