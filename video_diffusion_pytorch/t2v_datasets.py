import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import read_video

import random
import torch
from torchvision.io import read_video

class TextVideoDataset(Dataset):
    """テキスト（キャプション）と動画のペアデータセット"""

    def __init__(self, annotations_file, root_dir, transform=None, frame_count=10):
        """
        Args:
            annotations_file (string): path/to/annotations
            root_dir (string): path/to/videos
            transform (callable, optional): Optional transform to be applied on a sample.
            frame_count (int): Number of frames to load from each video
        """
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)['sentences']
        self.root_dir = root_dir
        self.transform = transform
        self.frame_count = frame_count
        self.video_ids = self._get_unique_video_ids()

    def _get_unique_video_ids(self):
        video_ids = list(set([x['video_id'] for x in self.annotations]))
        return video_ids

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        captions = [x['caption'] for x in self.annotations if x['video_id'] == video_id]
        # choose a random caption
        random_caption = random.choice(captions)
        video_filename = os.path.join(self.root_dir, f"{video_id}.mp4")

        video = self.load_video(video_filename)
        if self.transform:
            video = self.transform(video)

        sample = {'video_id': video_id, 'caption': random_caption, 'video': video}

        return sample
    
    def load_video(self, video_filename):
        video, _, _ = read_video(video_filename, start_pts=0, end_pts=None, pts_unit='sec')
        total_frames = video.shape[0]

        # Ensure there are enough frames to select from
        if total_frames >= self.frame_count:
            start_frame = random.randint(0, total_frames - self.frame_count)
            end_frame = start_frame + self.frame_count
            frames = video[start_frame:end_frame]
        else:
            # If there aren't enough frames, use what is available
            frames = video

        return frames


if __name__ == "__main__":
    annotations_file = './train_val_videodatainfo.json'  # path/to/annotations
    root_dir = './TrainValVideo'  # path/to/videos
    
    image_size = 64
    transform = T.Compose([
            T.Lambda(lambda x: x / 255.),
            T.Lambda(lambda x: x.permute(3, 0, 1, 2)),
            T.Resize(image_size),
            T.CenterCrop(image_size),
            ])

    dataset = TextVideoDataset(annotations_file=annotations_file, root_dir=root_dir, 
                               transform=transform, frame_count=10)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for i, sample in enumerate(dataloader):
        print(i, sample['video_id'], sample['caption'], sample['video'].shape)
        # 0 ['video541', 'video6134'] 
        # ['a cartoon truck travels through a snowy town', 'an ad for a movie is playing'] 
        # torch.Size([2, 3, 10, 64, 64])

