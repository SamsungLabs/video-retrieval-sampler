"""
Run this after you have the annotation files and all the videos downloaded.

Annotations are from Collaborative Experts: https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/msvd
and the raw-captions.pkl file is from http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/features-v2/MSVD-experts.tar.gz

Script outputs three json files for train, val, and test.
"""
import os
import json
import pickle
import argparse
from pathlib import Path
from utils import is_exist


def main(args, subset):

    with open(os.path.join(args.annot_dir, f"{subset}_list.txt"), "r") as f:
        video_ids = f.read().splitlines()
    
    with open(os.path.join(args.annot_dir, "raw-captions.pkl"), "rb") as f:
        captions = pickle.load(f)

    data = {}

    for video_id in video_ids:
        extension = is_exist(args.video_dir, video_id)
        if not extension:
            print(f"{video_id} does not exist.")
            continue
        data[video_id] = {"extension": extension}
        data[video_id]["sentences"] = [" ".join(cap) for cap in captions[video_id]]

    Path(os.path.join(args.output_folder)).mkdir(parents=True, exist_ok=True)

    print(f"Number of {subset} videos: {len(data)}")

    with open(os.path.join(args.output_folder, f"{subset}.json"), "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSVD annotation preprocessing")
    parser.add_argument('--video_dir', default='/mnt/sda/msvd/videos', help='path to downloaded videos')
    parser.add_argument('--annot_dir', default='/mnt/sda/msvd/annots', help='path to annotation files')
    parser.add_argument('--output_folder', default='annots/msvd')
    args = parser.parse_args()

    main(args, "train")
    main(args, "val")
    main(args, "test")