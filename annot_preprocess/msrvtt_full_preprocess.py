"""
Run this after you have the annotation files and all the videos downloaded.

Annotations are from CLIP4Clip: https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
and Collaborative Experts: https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/msrvtt

Script outputs three json files, for train, val, and test.
"""
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from utils import is_exist


def main(args, subset):

    with open(os.path.join(args.annot_dir, "full_split", f"{subset}_list_full.txt"), "r") as f:
        video_list = f.read().splitlines()

    with open(os.path.join(args.annot_dir, "MSRVTT_data.json"), "rb") as f:
        annot = json.load(f)

    data = {}

    for vid_annot in tqdm(annot["videos"]):
        video_id = vid_annot["video_id"]
        extension = is_exist(args.video_dir, video_id)
        if not extension:
            print(f"{video_id} does not exist.")
        if video_id in video_list:
            data[video_id] = {"extension": extension}
            data[video_id]["timestamps"] = [[vid_annot["start time"], vid_annot["end time"]]]
            data[video_id]["sentences"] = []

    for cap_annot in annot["sentences"]:
        video_id = cap_annot["video_id"]
        if video_id in video_list:
            data[video_id]["sentences"].append(cap_annot["caption"])

    Path(os.path.join(args.output_folder)).mkdir(parents=True, exist_ok=True)

    print(f"Number of {subset} videos: {len(data)}")

    with open(os.path.join(args.output_folder, f"{subset}.json"), "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ActivityNet annotation preprocessing")
    parser.add_argument('--video_dir', default='/mnt/sda/msrvtt/all_vids', help='path to downloaded videos')
    parser.add_argument('--annot_dir', default='/mnt/sda/msrvtt/clip4clip_annots', help='path to annotation files')
    parser.add_argument('--output_folder', default='annots/msrvtt-full')
    args = parser.parse_args()

    main(args, "train")
    main(args, "val")
    main(args, "test")