# SSM-Meets-Video-Diffusion-Models

"SSM Meets Video Diffusion Models: Efficient Long-Term Video Generation with Selective State Spaces" [[Paper]](https://arxiv.org/abs/2403.07711)

![Image1](images/Figure1.png)

![MineRL](images/figure4.png)


## Devices
We use NVIDIA A100 $\times 8$ for training.  
All models were trained for 100k steps, and training for all configurations in the paper was completed within four days.

## Settings
Please use `./Dockerfile` to build docker image or install python libraries specified in this dockerfile.

## Run Experimental Codes

### Downloading Datasets
#### MineRL Navigate
1. Execute a following python code.
```
python dl_mine_rl.py
```
2. Specify `minerl` as `--dataset`, and `minerl_navigate-torch` as `--folder`.

#### GQN-Mazes


### Training
```
python train_video-diffusion.py 
--timesteps 256 --loss_type 'l2' --train_lr 0.0003 --train_num_steps 700000 --train_batch_size 16 --gradient_accumulate_every 2 --ema_decay 0.995 # Learning Settings
--base_channel_size 64 --timeemb_linears 2 # Architecture Settings
--temporal_layer 'bi-s4d' --s4d_version 8 # Temporal Layer Settings
--image_size 32 --dataset 'ucf101-all' # Dataset Settings
--folder 'path/to/datasets' 
--results_folder 'path/to/save' 
--device_ids 0 1 2 3 # GPU Settings
```
### Sampling
```
python sample_video-diffusion.py 
--timesteps 256 --loss_type 'l2' --train_lr 0.0003 --train_num_steps 700000 --train_batch_size 16 --gradient_accumulate_every 2 --ema_decay 0.995 # Learning Settings
--base_channel_size 64 --timeemb_linears 2 # Architecture Settings
--temporal_layer 'bi-s4d' --s4d_version 8 # Temporal Layer Settings
--image_size 32 --dataset 'ucf101-all' # Dataset Settings
--folder 'path/to/datasets' 
--results_folder 'path/to/save'
--num_samples 2500 --sample_batch_size 10 --sample_save_every 10 # Sampling Number Settings
--milestone 92                                                   # Sampling Milestone (Progress of Learning) Settings
--device_ids 0 --seed 0                                          # Sampling Device Settings
```
### Evaluation
```
python eval_video-diffusion.py 
--timesteps 256 --loss_type 'l2' --train_lr 0.0003 --train_num_steps 700000 --train_batch_size 16 --gradient_accumulate_every 2 --ema_decay 0.995 # Learning Settings
--base_channel_size 64 --timeemb_linears 2 # Architecture Settings
--temporal_layer 'bi-s4d' --s4d_version 8 # Temporal Layer Settings
--image_size 32 --dataset 'ucf101-all' # Dataset Settings
--folder 'path/to/datasets' 
--results_folder 'path/to/save'
--num_samples 2500 --sample_batch_size 10 --sample_save_every 10 
--milestone 92                                                   
# --seed 0 --sample_seeds 0 1 2 3 --eval_batch_size 100 # Evaluation Settings
```

## Citation

```bibtex
@misc{ssmvdm2024,
      title={SSM Meets Video Diffusion Models: Efficient Video Generation with Structured State Spaces}, 
      author={Yuta Oshima and Shohei Taniguchi and Masahiro Suzuki and Yutaka Matsuo},
      year={2024},
      eprint={2403.07711},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
