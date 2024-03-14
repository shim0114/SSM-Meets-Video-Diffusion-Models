# SSM-Meets-Video-Diffusion-Models

## Training Details
| Dataset          | UCF101         | UCF101         | MineRL         | MineRL         |
|------------------|----------------|----------------|----------------|----------------|
| **# of Frames**  | 16             | 16             | 64             | 150            |
| **Resolution**   | $32 \times 32$ | $64 \times 64$ | $32 \times 32$ | $32 \times 32$ |
| **Base channel size** | 64       | 64             | 64             | 64             |
| **Channel multipliers** | 1, 2, 4, 8 | 1, 2, 4, 8 | 1, 2, 4, 8 | 1, 2, 4, 8 |
| **Time embedding dimension** | 1024 | 1024 | 1024 | 1024 |
| **Time embedding linears** | 2 | 2 | 2 | 2 |
| **# of attention heads (for attentions)** | 8 | 8 | 8 | 8 |
| **Dims of attention (for attentions)** | 64 | 64 | 64 | 64 |
| **SSM hidden dims (for SSMs)** | 512 | 512 | 512 | 512 |
| **MLP hidden dims (for SSMs)** | 512 | 512 | 512 | 512 |
| **Denoising timesteps (T)** | 256 | 1000 | 256 | 256 |
| **Loss type** | L2 loss of \( \epsilon \) | L2 loss of \( \epsilon \) | L2 loss of \( \epsilon \) | L2 loss of \( \epsilon \) |
| **Training steps** | 92k | 106k | 174k | 129k |
| **Optimizer** | Adam | Adam | Adam | Adam |
| **Training learning rate** | 0.0003 | 0.0001 | 0.0003 | 0.0003 |
| **Train batch size** | 32 | 32 | 8 | 8 |
| **EMA decay** | 0.995 | 0.995 | 0.995 | 0.995 |
| **GPUs** | V100 $\times 4$ | A100 $\times 8$ | V100 $\times 4$ | V100 $\times 4$ |
| **Training Time** | 72 hours | 120 hours | 72 hours | 72 hours |

## Settings
Please use `./Dockerfile` to build docker image or install python libraries specified in this dockerfile.

## Run Experimental Codes

### Training
```
python train_video-diffusion.py 
--timesteps 256 --loss_type 'l2' --train_lr 0.0003 --train_num_steps 700000 --train_batch_size 16 --gradient_accumulate_every 2 --ema_decay 0.995 # Learning Settings
--base_channel_size 64 --timeemb_linears 2 # Architecture Settings
--temporal_layer 'bi-s4d' --s4d_version 16 # Temporal Layer Settings
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
--temporal_layer 'bi-s4d' --s4d_version 16 # Temporal Layer Settings
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
--temporal_layer 'bi-s4d' --s4d_version 16 # Temporal Layer Settings
--image_size 32 --dataset 'ucf101-all' # Dataset Settings
--folder 'path/to/datasets' 
--results_folder 'path/to/save'
--num_samples 2500 --sample_batch_size 10 --sample_save_every 10 
--milestone 92                                                   
# --seed 0 --sample_seeds 0 1 2 3 --eval_batch_size 100 # Evaluation Settings
```