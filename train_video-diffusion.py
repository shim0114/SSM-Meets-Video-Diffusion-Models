import torch
from video_diffusion_pytorch.wip_video_diffusion_pytorch import Unet3D, GaussianDiffusion, Trainer
import os
import re
import argparse
import wandb

def parse_arguments():
    parser = argparse.ArgumentParser(description='Video Diffusion Training')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--channels', type=int, default=3, help='Image Channel size') 
    parser.add_argument('--num_frames', type=int, default=16, help='Number of video frames')
    parser.add_argument('--base_channel_size', type=int, default=256, help='Base channel size of 3D U-Net')
    parser.add_argument('--timeemb_linears', type=int, default=2, help='Number of temporal layers')
    parser.add_argument('--attn_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--attn_dim_head', type=int, default=64, help='Dimension of attention head')
    parser.add_argument('--ssm_hidden_dim', type=int, default=16, help='Hidden dimension of SSM')
    parser.add_argument('--ssm_linear_dim', type=int, default=None, help='Linear dimension of MLPs in SSM Layers')
    parser.add_argument('--ssm_version', type=int, default=None, help='Version of SSM Layers')
    parser.add_argument('--temporal_layer', type=str, default='attn', help='Architecture of temporal layers') 
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of steps') 
    parser.add_argument('--loss_type', type=str, default='l2', help='Loss type')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--train_lr', type=float, default=1e-4, help='Training learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam')
    parser.add_argument('--save_and_sample_every', type=int, default=1000, help='Save and sample frequency') 
    parser.add_argument('--train_num_steps', type=int, default=700000, help='Total training steps') 
    parser.add_argument('--gradient_accumulate_every', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--ema_decay', type=float, default=0.995, help='Exponential moving average decay')
    parser.add_argument('--amp', type=bool, default=False, help='Use mixed precision')
    parser.add_argument('--dataset', type=str, default='movingmnist', help='Dataset name')
    parser.add_argument('--folder', type=str, default='/groups/gcb50389/yuta.oshima/video_datasets/mnist_test_seq.npy', help='Data folder')
    parser.add_argument('--results_folder', type=str, default='/groups/gcb50389/yuta.oshima/s5_diffusion_results/results', help='Results folder')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0, 1, 2], help='List of GPU IDs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_cond', type=bool, default=False, help='Use text condition or not')
    parser.add_argument('--cond_folder', type=str, default=None, help='Text condition folder')
    args = parser.parse_args()
    return args

def main(args):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    
    wandb_run = wandb.init(
        project='iclr_ssm_diffusion', 
        entity="shim0114", 
        config=vars(args))

    model = Unet3D(
        dim = args.base_channel_size, # 256 -> 128
        dim_mults = (1, 2, 4, 8),
        channels = args.channels, 
        timeemb_linears = args.timeemb_linears,
        attn_heads = args.attn_heads,
        attn_dim_head = args.attn_dim_head,
        ssm_hidden_dim = args.ssm_hidden_dim,
        ssm_linear_dim = args.ssm_linear_dim,
        ssm_version = args.ssm_version,
        temporal_arch = args.temporal_layer,
    )
    
    from thop import profile 
    img = torch.randn(1, args.channels, args.num_frames, args.image_size, args.image_size).to('cuda:0')
    t = torch.tensor([1]).to('cuda:0')
    flops, params = profile(model.to('cuda:0'), inputs=(img, t))

    print(params)
    print(flops)
    wandb_run.log({"params": params, "flops": flops})

    model = model.cpu()
    
    model = torch.nn.DataParallel(model, device_ids=args.device_ids) # DP (old ver)

    diffusion = GaussianDiffusion(
        model,
        image_size = args.image_size, 
        channels = args.channels, 
        num_frames = args.num_frames, 
        timesteps = args.timesteps,   # number of steps
        loss_type = args.loss_type    # L1 or L2
    )
    
    diffusion = diffusion.cuda()
    # diffusion = torch.nn.DataParallel(diffusion, device_ids=args.device_ids) # DP (new ver)
   
    trainer = Trainer(
        diffusion,
        dataset=args.dataset, # dataset name
        folder=args.folder, # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
        results_folder = args.results_folder, 
        use_cond = args.use_cond,
        cond_folder = args.cond_folder,
        train_batch_size = args.train_batch_size, 
        train_lr = args.train_lr, 
        beta1 = args.beta1,
        beta2 = args.beta2,
        save_and_sample_every = args.save_and_sample_every,
        train_num_steps = args.train_num_steps, # total training steps 
        gradient_accumulate_every = args.gradient_accumulate_every, # gradient accumulation steps
        ema_decay = args.ema_decay,             # exponential moving average decay
        amp = args.amp,                         # turn on mixed precision
        num_gpus = len(args.device_ids),        # number of GPUs
    )
    
    if args.resume_milestone > 0:
        # resume training from last saved milestone   
        trainer.load(args.resume_milestone)
        print(args.resume_milestone)

    trainer.train(log_fn=wandb_run.log)  
    
if __name__ == "__main__":
    args = parse_arguments()
    
    # resume training from maximum milestone 
    if os.path.exists(args.results_folder):
        # search maximum number of saved milestions
        pattern = re.compile(r'model-(\d+)\.pt$')
        max_num = -1
        for filename in os.listdir(args.results_folder):
            match = pattern.match(filename)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num       
        args.resume_milestone = max_num
    else:
        args.resume_milestone = -1
        
    main(args)