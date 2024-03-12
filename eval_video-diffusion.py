from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import numpy as np
import os
import glob
import re
import argparse

import torch
from torch.utils import data
from torchvision.datasets import UCF101
from torchvision import transforms as T, utils

from tqdm import tqdm
import wandb

from video_diffusion_pytorch.long_video_datasets import MineRLDataset, GQNMazesDataset, CarlaDataset ### Changed ###
from video_diffusion_pytorch.wip_video_diffusion_pytorch import custom_video_collate_fn
from frechet_video_distance import frechet_video_distance as fvd
from frechet_video_distance.util import open_url

def parse_arguments():
    parser = argparse.ArgumentParser(description='Video Diffusion Evaluation')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--channels', type=int, default=3, help='Image Channel size') 
    parser.add_argument('--num_frames', type=int, default=16, help='Number of video frames')
    parser.add_argument('--base_channel_size', type=int, default=256, help='Base channel size of 3D U-Net')
    parser.add_argument('--timeemb_linears', type=int, default=2, help='Number of temporal layers')
    parser.add_argument('--attn_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--attn_dim_head', type=int, default=64, help='Dimension of attention head')
    parser.add_argument('--s4d_hidden_dim', type=int, default=None, help='Hidden dimension of S4D')
    parser.add_argument('--s4d_linear_dim', type=int, default=None, help='Linear dimension of S4D')
    parser.add_argument('--s4d_version', type=int, default=None, help='Version of S4D')
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
    parser.add_argument('--folder', type=str, help='Data folder')
    parser.add_argument('--results_folder', type=str, help='Results folder')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples')
    parser.add_argument('--sample_batch_size', type=int, default=16, help='Sampling batch size')
    parser.add_argument('--sample_save_every', type=int, default=100, help='Sample save frequency')
    parser.add_argument('--milestone', type=int, help='Milestone for loading model')
    parser.add_argument('--sample_seeds', nargs='+', type=int, default=[0, 1, 2, 3], help='List of sample seeds')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Evaluation batch size')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0, 1, 2], help='List of GPU IDs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # parser.add_argument('--csv_path', type=str, default="/home/acd13972py/Video-LLaMA/video_captions.csv", help='CSV path')
    # parser.add_argument('--use_cond', type=bool, default=True, help='use text condition or not')
    args = parser.parse_args()
    return args

def preprocess(videos, target_resolution):
    """Runs some preprocessing on the videos for I3D model.

    Args:
        videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
        preprocessed. We don't care about the specific dtype of the videos, it can
        be anything that tf.image.resize_bilinear accepts. 
        Values are expected to be in the range 0-255. <- ### Changed ### [-1, 1]
        target_resolution: (width, height): target video resolution

    Returns:
        videos: <float32>[batch_size, num_frames, height, width, depth]
    """
    videos_shape = videos.shape.as_list()
    all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
    resized_videos = tf.image.resize_bilinear(all_frames, size=target_resolution)
    target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
    output_videos = tf.reshape(resized_videos, target_shape)
    # scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
    return output_videos # scaled_videos

def compute_acts(videos: np.ndarray) -> float:
    
    with tf.Graph().as_default():
        
        videos = tf.convert_to_tensor(videos, np.float32)
        videos = preprocess(videos, (224, 224))
        activations = fvd.create_id3_embedding(videos)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            return sess.run(activations)    
        
def main(args):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    
    wandb_run = wandb.init(
        project='ssm_vdm_sampling', 
        config=vars(args))
    
    if args.dataset == 'ucf101-all':
        # if args.num_frames != 16:
        #     raise ValueError('UCF101 dataset should be 16 frames')
        ds_train = UCF101(
            root=args.folder + '/UCF-101',
            annotation_path=args.folder + '/ucfTrainTestlist',
            frames_per_clip=args.num_frames,
            step_between_clips=100000, # for evaluation, we use each video only once
            train=True,
            transform=T.Compose([
        T.Lambda(lambda x: x / 255.),
        T.Lambda(lambda x: x.permute(3, 0, 1, 2)),
        T.Resize(args.image_size),
        T.CenterCrop(args.image_size),
        ])
        )
        ds_test = UCF101(
            root=args.folder + '/UCF-101',
            annotation_path=args.folder + '/ucfTrainTestlist',
            frames_per_clip=args.num_frames,
            step_between_clips=100000, # for evaluation, we use each video only once
            train=False,
            transform=T.Compose([
        T.Lambda(lambda x: x / 255.),
        T.Lambda(lambda x: x.permute(3, 0, 1, 2)),
        T.Resize(args.image_size),
        T.CenterCrop(args.image_size),
        ])
        )
        ds = data.ConcatDataset([ds_train, ds_test])
    elif args.dataset == 'minerl':
        ds = MineRLDataset(
            path=args.folder, 
            shard=0, 
            num_shards=1, 
            T=args.num_frames,
            image_size = args.image_size
        )
    else:
        raise ValueError('dataset not implemented')

    print(f'found {len(ds)} videos as gif files at {args.folder}')
    assert len(ds) > 0, 'need to have at least 1 video to start training (although 1 is not great, try 100k)'

    if args.dataset == 'ucf101-all':
        dl = data.DataLoader(ds, batch_size = args.eval_batch_size, shuffle=True, pin_memory=True, drop_last=False, collate_fn=custom_video_collate_fn)
    elif args.dataset == 'minerl':
        dl = data.DataLoader(ds, batch_size = args.eval_batch_size, shuffle=True, pin_memory=True, drop_last=False)
    else:
        raise ValueError('dataset not implemented')
    
    all_reals_activation = []
    for reals_video in tqdm(dl):
        if args.dataset == 'ucf101-all':
            reals_video = reals_video
            reals_video = reals_video.permute(0, 2, 3, 4, 1)
        elif args.dataset == 'minerl':
            reals_video = reals_video[0]
            reals_video = reals_video.permute(0, 1, 3, 4, 2)
        reals_video = reals_video.numpy()
        reals_activation = compute_acts(reals_video)
        all_reals_activation.extend(reals_activation)

    all_fakes_activation = []
    all_fakes_path = glob.glob(os.path.join(args.results_folder, f'fakes/{args.milestone}_*'))
    for fakes_path in tqdm(all_fakes_path):
        
        # 使用するサンプルの選択(サンプリング時のcrash対応のための余分の排除)
        match = re.search(r'fakes/\d+_(\d+)+_(\d+).npy', fakes_path)
        sample_seed = match.group(1)
        number_a = match.group(2)
        if int(sample_seed) in args.sample_seeds:
            if not int(number_a) > args.num_samples:
                fakes_video = np.load(fakes_path)
                fakes_video = fakes_video.transpose(0, 2, 3, 4, 1)
                fakes_activation = compute_acts(fakes_video)
                all_fakes_activation.extend(fakes_activation)
            
    print('<==============================================================>')
    print(f'[reals_activation]: {len(all_reals_activation)}.')
    print(f'[fakes_activation]: {len(all_fakes_activation)}.')
    print('<==============================================================>')

    print('<==============================================================>')
    print('Computing FVD...')
    print('<==============================================================>')
    fvd_result = fvd.calculate_fvd(all_reals_activation, all_fakes_activation)

    print('<==============================================================>')
    print(f'[FVD scores]: {fvd_result}.')
    print('<==============================================================>')
    
    wandb_run.log({args.dataset + '/total_reals': len(all_reals_activation),
                   args.dataset + '/total_fakes': len(all_fakes_activation),
                   args.dataset + '/FVD': fvd_result}) 


if __name__ == "__main__":
    args = parse_arguments()
    main(args)