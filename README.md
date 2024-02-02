# Self-supervised Learning of Semantic Correspondence Using Web Videos

[Our paper](https://openaccess.thecvf.com/content/WACV2024/papers/Kwon_Self-Supervised_Learning_of_Semantic_Correspondence_Using_Web_Videos_WACV_2024_paper.pdf), [Project page](https://cvlab.postech.ac.kr/research/SSSCWEB/)

*Donghyeon Kwon, Minsu Cho and Suha Kwak*

> kinux98@postech.ac.kr

This repository contains the official implementation of : 
> Self-supervised Learning of Semantic Correspondence Using Web Videos
> 
that has been accepted to 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2024).


<div align="center">
  <img src="semi_step.png"/>
</div>

## Highlights

• We present the first attempt to utilize web videos for learning semantic correspondence in a self-supervised learning manner.

• We provide a fully automatic process for dataset construction and labeling using web videos. Our strategy exploits the exclusive advantages of videos over images for generating accurate pseudo correspondence labels.

• Our method outperformed existing self-supervised learning models and even substantially improved supervised learning performance through transfer learning.

## Requirements
The repository is tested on Ubuntu 20.04.1 LTS, Python 3.8.16, PyTorch 2.0.1+cu117. We use four NVIDIA RTX 6000 Ada Generation (49GB for each) for training.

After preparing virtual environment, download requirementes packages with : 

> pip install requirements.txt

or

> conda --name SSSCWEB --file requirements_conda_ver.txt

### wandb settings

Before starting, you should login wandb using your personal API key.

>wandb login PERSONAL_API_KEY


# Data preparation

## Web video download

Before started, you have to download youtube videos for generating pseudo correspondence labels with following link : [download](https://postechackr-my.sharepoint.com/:u:/g/personal/kinux98_postech_ac_kr/EQDZT5o3OTxPmqr0gbKCgBgBVWly44pl_5FY4C1cfIFrPA?e=GVNxjL), total 46.5GB.

or 

you may download web videos by yourself with provided codes in youtube_download folder. 

To do this, you may change `class_list`, `data_range_start/end` in `common.py`

And then run ``GetURL.py``. It will downlaod searched video youtube-id (not a video!) and their meta-info with thumbnail image. 

Finally, run `youtube_downloader.py` with proper path. It will automatically download youtube videos from provided meta-info.


## Video frame extraction

After downloading videos, you have to extract frames of each video. To do this, you have to run `extract_shot_multi.py` in video_preprocess folder. You may change `root`, `video_dir` and `image_dir` to proper path. You can also change `process_num` with proper value (it will extract each video's frame in multi-threading manner).


(If you have downloaded youtube videos by yourself, please prepare your own version of parsing.json file. You can make it by uncomment L91-92 in `extract_shot_multi.py` in video_preprocess folder)


## Pseudo label generation

Now we have extracted frames and it's time to generate pseudo correspondence labels. 

Run 

`
python youtube_mp.py --resume=/your/path/checkpoints/youtube_consecutive/checkpoint.pth
`

in frame_preprocess folder (you may change `root`, `json_file`, `image_set` and `video_which` to proper path in `youtube_mp.py`)

It will generate pseudo correspondence labels in results folder. 

Note that we have already provided `video_scene_parsiong_new_mt.json` for our videos. 


# Training
Before training, please prepare Spair-71K, PF-PASCAL and PF-WILLOW for yourself.

For SPair-71K, run

`
python train.py --snapshots=./snapshots/unsup_spair --run_yt=True --run_sb=False --run_dann=True --run_contra=False --benchmark=spair --eval_benchmark=spair --feature-size=24
`

For PF-PASCAL and PF-WILLOW, run

`
python train.py --snapshots=./snapshots/unsup_pfpascal_pfwillow --run_yt=True --run_sb=False --run_dann=True --run_contra=False --benchmark=pfpascal --eval_benchmark=pfpascal --eval_benchmark2=pfwillow
`

## Citation
If you find this project useful, please consider citing as follows:
```
@InProceedings{Kwon_2024_WACV,
    author    = {Kwon, Donghyeon and Cho, Minsu and Kwak, Suha},
    title     = {Self-Supervised Learning of Semantic Correspondence Using Web Videos},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {2142-2152}
}
```

### Acknowledgements

We borrow some codes from 

 - https://github.com/SunghwanHong/Cost-Aggregation-transformers

 - https://github.com/ajabri/videowalk