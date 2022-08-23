import os
import json
import glob
import torch
import random
import numpy as np
from PIL import Image
from utils.logger import LOGGER
from modeling.clip import tokenize
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets.preprocess import get_transforms


class BaseDataset(Dataset):

    def __init__(self, cfg, annot_file, is_train):
        self.frames_dir = cfg.frames_dir
        self.img_tmpl = cfg.img_tmpl
        self.concat_captions = cfg.concat_captions
        self.max_txt_len = cfg.max_txt_len
        self.num_frm = cfg.num_frm
        self.num_frm_subset = cfg.num_frm_subset
        self.sampling = cfg.sampling
        self.max_img_size = cfg.max_img_size
        self.multi_cap_eval = cfg.multi_cap_eval
        self.is_train = is_train

        self.clip_preprocess = get_transforms("clip", cfg.max_img_size)
        self.policy_preprocess = None
        if "clip" not in cfg.backbone:
            if cfg.backbone == "raw":
                image_size = cfg.rescale_size
            elif cfg.backbone.startswith("efficientnet"):
                from efficientnet_pytorch import EfficientNet
                image_size = EfficientNet.get_image_size(cfg.backbone)
                image_size = 224 if "msrvtt" in annot_file else image_size
            else: # mobilenet_v2
                image_size = 224 if "msrvtt" in annot_file else 256
            self.policy_preprocess = get_transforms(cfg.backbone, image_size)

        if self.num_frm_subset > 0:
            self.num_frm_subset = min(self.num_frm, self.num_frm_subset)

        with open(annot_file, 'r') as f:
            self.annot = json.load(f)

        self.vid_query_pairs = self.prepare_vid_query_pairs()

        self.sample_len = len(self.vid_query_pairs)
        if cfg.data_subset:
            self.sample_len = min(self.sample_len, cfg.data_subset)

    def prepare_vid_query_pairs(self):
        vid_query_pairs = []
        if self.concat_captions == "concat": # ActivityNet, DiDeMo
            for vid in self.annot:
                vid_query_pairs.append((vid, ' '.join(self.annot[vid]['sentences'])))
        elif self.concat_captions.startswith("indep"): # MSRVTT, MSVD
            if (not self.is_train and self.multi_cap_eval) or self.concat_captions == "indep-one":
                for vid in self.annot:
                    vid_query_pairs.append((vid, self.annot[vid]['sentences']))
            elif self.concat_captions == "indep-all":
                for vid in self.annot:
                    for s in self.annot[vid]['sentences']:
                        vid_query_pairs.append((vid, s))
        return vid_query_pairs

    def __len__(self):
        return self.sample_len

    def sample_images(self, vid):
        num_frames = len(glob.glob1(os.path.join(self.frames_dir, vid),"*.jpg"))
        if not num_frames:
            return [], []
        indices = np.linspace(0, num_frames, num=self.num_frm, endpoint=False, dtype=int) # Uniform sampling
        if self.num_frm_subset > 0: # ablation study
            if self.sampling == "random":
                indices = np.random.choice(indices, size=self.num_frm_subset, replace=False)
                indices = np.sort(indices) # keep temporal order
            elif self.sampling == "uniform":
                indices_subset = np.linspace(0, len(indices), num=self.num_frm_subset, endpoint=False, dtype=int)
                indices = indices[indices_subset]
            else:
                raise ValueError(f"Invalid frame sampling method {self.sampling}.")
        clip_images = []
        policy_images = []
        for img_idx in indices:
            filename = os.path.join(self.frames_dir, vid, self.img_tmpl.format(img_idx+1))
            im = Image.open(filename)
            clip_images.append(self.clip_preprocess(im)) # 3, 224, 224
            if self.policy_preprocess is not None:
                policy_images.append(self.policy_preprocess(im))
        if policy_images:
            return clip_images, policy_images
        else:
            return clip_images, clip_images

    def __getitem__(self, idx):
    
        # skip error videos:
        num_retries = 3
        for _ in range(num_retries):
            vid, caption = self.vid_query_pairs[idx]

            if isinstance(caption, list) and self.is_train:
                caption = np.random.choice(caption)

            if isinstance(caption, str):
                caption = [caption]

            tokens = tokenize(caption)

            tokens_fixed = torch.zeros((len(tokens), self.max_txt_len), dtype=torch.int64)
            for i in range(len(tokens)):
                if len(tokens[i]) > self.max_txt_len:
                    tokens[i] = tokens[i][:self.max_txt_len-1] + tokens[i][-1:]
                tokens_fixed[i, :len(tokens[i])] = torch.tensor(tokens[i])

            clip_images, policy_images = self.sample_images(vid)
            if not clip_images:
                LOGGER.info(f"Failed to load examples with video: {vid}.")
                if not self.is_train:
                    return None
                LOGGER.info("Will randomly sample an example as a replacement.")
                idx = random.randint(0, len(self) - 1)
                continue
            clip_images = torch.stack(clip_images)
            policy_images = torch.stack(policy_images)

            sample = {
                "clip_images": clip_images,
                "policy_images": policy_images,
                "tokens": tokens_fixed,
            }
            return sample
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

    def collate_data(self, batch):

        batch = [b for b in batch if b is not None]
        clip_images = pad_sequence([d["clip_images"] for d in batch], batch_first=True, padding_value=0)
        policy_images = pad_sequence([d["policy_images"] for d in batch], batch_first=True, padding_value=0)
        tokens = pad_sequence([d["tokens"] for d in batch], batch_first=True, padding_value=0)

        collated_batch = dict(
            clip_inputs=clip_images,
            policy_inputs=policy_images,  # (B, #frm, C, H, W)
            text_input_ids=tokens,
        )

        return collated_batch
