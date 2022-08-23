# mmSampler: Efficient Frame Sampler for Multimodal Video Retrieval

We study the problem of natural language-based video retrieval, the task of finding relevant videos given natural language search queries. Most recent state-of-the-art (SOTA) approaches would encode the video and query separately and map the video and query embeddings into a joint latent space to calculate a similarity score between them. To learn a video representation, existing solutions either use all the frames or sample a subset of frames from the video using uniform sampling. The former solution could be computationally prohibitive while the latter may inject noise from uninformative frames into the final video representation. To this end, we propose mmSampler, a learning-based sampler, to adaptively select salient frames to represent the videos for multimodal video retrieval. mmSampler can greatly reduce the computational overhead for video representation without affecting the retrieval performance. We learn a lightweight policy network to decide whether to further process or discard a frame given the current and past frame observations. By adopting the Gumbel-Softmax trick, we avoid backpropagating through undefined gradients and train the sampler end-to-end in an efficient manner. Experimental results on benchmark datasets such as ActivityNet, DiDeMo and MSRVTT demonstrate that mmSampler achieves strong retrieval performance while saving as much as 43\% GFLOPs per video.

This repo is the implementation for the paper "mmSampler: Efficient Frame Sampler for Multimodal Video Retrieval".

## Maintainers
- Angela Ye (angela.ye@samsung.com)
- Zhiming Hu (zhiming.hu@samsung.com)

## Prerequisites

- Linux (Ubuntu 16.04 or later is recommended)
- Python 3.7
- Packages:
    - ffmpeg (`$sudo apt-get install ffmpeg`)
- Datasets: [ActivityNet Dense Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/), [MSRVTT](http://ms-multimedia-challenge.com/2017/dataset), [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/), [DiDeMo](https://github.com/LisaAnne/LocalizingMoments)

## How to Install

Create a conda environment and install the appropriate packages:
```
$ conda create -n mmsampler python=3.7 -y
$ conda activate mmsampler
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install tqdm tensorboard easydict ftfy regex ptflops efficientnet_pytorch pandas mlflow
```
Or try the script
```
$ source scripts/install_packages.sh
```

Update `cudatoolkit=11.0` with the appropriate CUDA version on your machine.

## Datasets

### MSRVTT

The videos are shared by [Frozen in Time](https://github.com/m-bain/frozen-in-time#-finetuning-benchmarks-msr-vtt):
```
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
```

### DiDeMo

The videos can be downloaded from [LisaAnne/LocalizingMoments](https://github.com/LisaAnne/LocalizingMoments).

### ActivityNet

Download the videos from the [official website](http://activity-net.org/download.html). The authors have made the videos available on Google and Baidu drives.


## Preprocessing

### Frame Extraction

Run `frame_extraction.py` after having downloaded the dataset videos and annotations from the website. Make sure that all the videos are in the same directory (no sub-directories allowed).

```
python frame_extraction.py /path/to/videos /path/to/frames --parallel
```

Subsequently, update the `frames_dir` parameter in the config files `configs/[dataset].json`.

### Annotation Preprocessing

If the videos downloaded differ from the set used in the paper, run `annot_preprocess/{dataset}_preprocess.py` to generate train/test splits used by the dataloader. Splits used in the paper can be found in `annots/`.

To obtain the annotation files used to generate the splits, please download them from the following links:
- MSRVTT annotations are from [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip): 
```
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip
```
- ActivityNet annotations are from the [project page](https://cs.stanford.edu/people/ranjaykrishna/densevid/) of ActivityNet Captions:
```
wget https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip
```
- DiDeMo annotations have two components: annotations from the [original author](https://github.com/LisaAnne/LocalizingMoments/tree/master/data) and the split used by [Collaborative Experts](https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/didemo).

## Training

Train a policy network on ActivityNet
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/activitynet.json --frames_dir path/to/frames --uniform_weight 0.03 --backbone mobilenet_v2 --freeze_cnn --num_workers 16
```

Train a policy network on DiDeMo
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/didemo.json --frames_dir path/to/frames --uniform_weight 0.03 --backbone mobilenet_v2 --freeze_cnn --num_epochs 5
```

Train a policy network on MSRVTT
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/msrvtt-jsfusion.json --frames_dir path/to/frames --uniform_weight 0.03 --backbone mobilenet_v2 --freeze_cnn
```

To train using feature differences, add a `--diff` flag. To train using feature concatenation, add a `--concat` flag. To replace the 1-layer transformer with an LSTM, add a `--rnn lstm` flag.

Train a video retrieval model **without a policy** on MSRVTT
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --config configs/msrvtt-jsfusion.json --frames_dir path/to/frames --no_policy
```

You may also generate all the main results using `main_results.sh`:
```
chmod +x scripts/main_results.sh
./scripts/main_results.sh ${NUM_GPUS}
```

You may need to update the batch sizes in the config files and/or the number of GPUs used by changing the `--nproc_per_node=num_gpu` parameter.

## Testing

Inference on MSRVTT with a policy network
```
python -m torch.distributed.launch --nproc_per_node=1 train.py --config configs/msrvtt-jsfusion.json --frames_dir path/to/frames --backbone mobilenet_v2 --do_inference --resume path/to/trained/model
```

Inference on MSRVTT **without a policy**
```
python -m torch.distributed.launch --nproc_per_node=1 train.py --config configs/msrvtt-jsfusion.json --frames_dir path/to/frames --no_policy --do_inference --resume path/to/trained/model
```

## Results

Video-to-text retrieval on ActivityNet: 
| Model | # Frames used | GFLOPs/v | R@1 | R@5 | R@10 | MdR | MnR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NoPolicy | 32 | 141.12 | 41.3 | 72.3 | 84.0 | 2 | 7.5 |
| Ours-frame | 15.88 | 80.32 | 42.0 | 72.4 | 84.1 | 2 | 7.4 |
| Ours-diff | 15.76 | 79.84 | 42.2 | 72.2 | 83.8 | 2 | 7.6 |
| Ours-concat | 15.99 | 80.96 | 42.7 | 72.9 | 84.6 | 2 | 7.2 |

Text-to-video retrieval on ActivityNet: 
| Model | # Frames used | GFLOPs/v | R@1 | R@5 | R@10 | MdR | MnR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NoPolicy | 32 | 141.12 | 42.1 | 74.0 | 85.1 | 2 | 7.0 |
| Ours-frame | 15.88 | 80.32 | 43.7 | 74.4 | 85.8 | 2 | 7.0 |
| Ours-diff | 15.76 | 79.84 | 43.0 | 74.0 | 85.1 | 2 | 7.2 |
| Ours-concat | 15.99 | 80.96 | 44.0 | 74.5 | 85.7 | 2 | 6.8 |

Video-to-text retrieval on DiDeMo: 
| Model | # Frames used | GFLOPs/v | R@1 | R@5 | R@10 | MdR | MnR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NoPolicy | 32 | 141.12 | 40.7 | 68.9 | 79.1 | 2 | 18.6 |
| Ours-frame | 19.16 | 94.88 | 41.4 | 70.1 | 80.0 | 2 | 18.3 |
| Ours-diff | 16.78 | 84.32 | 41.3 | 69.0 | 80.3 | 2 | 18.7 |
| Ours-concat | 16.69 | 84.00 | 41.6 | 70.0 | 79.7 | 2 | 18.3 |

Text-to-video retrieval on DiDeMo: 
| Model | # Frames used | GFLOPs/v | R@1 | R@5 | R@10 | MdR | MnR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NoPolicy | 32 | 141.12 | 41.0 | 69.1 | 78.9 | 2 | 11.5 |
| Ours-frame | 19.16 | 94.88 | 41.8 | 70.9 | 80.5 | 2 | 11.0 |
| Ours-diff | 16.78 | 84.32 | 42.0 | 69.8 | 79.6 | 2 | 11.3 |
| Ours-concat | 16.69 | 84.00 | 41.7 | 71.2 | 79.3 | 2 | 11.4 |

Video-to-text retrieval on MSRVTT: 
| Model | # Frames used | GFLOPs/v | R@1 | R@5 | R@10 | MdR | MnR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NoPolicy | 16 | 70.56 | 42.2 | 68.7 | 79.2 | 2 | 16.5 |
| Ours-frame | 11.03 | 53.84 | 43.7 | 71.2 | 79.8 | 2 | 15.6 |
| Ours-diff | 9.67 | 47.84 | 42.8 | 70.6 | 80.6 | 2 | 15.6 |
| Ours-concat | 10.09 | 49.68 | 43.6 | 70.6 | 80.1 | 2 | 15.8 |

Text-to-video retrieval on MSRVTT: 
| Model | # Frames used | GFLOPs/v | R@1 | R@5 | R@10 | MdR | MnR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NoPolicy | 16 | 70.56 | 42.1 | 70.4 | 81.2 | 2 | 11.7 |
| Ours-frame | 11.03 | 53.84 | 44.4 | 71.5 | 82.1 | 2 | 10.6 |
| Ours-diff | 9.67 | 47.84 | 43.9 | 71.3 | 82.4 | 2 | 10.9 |
| Ours-concat | 10.09 | 49.68 | 44.3 | 72.1 | 82.1 | 2 | 11.2 |

Note NoPolicy is essentially the [CLIP4Clip](https://arxiv.org/abs/2104.08860) implementation.

## Citation
```
@inproceedings{hu2022mmsampler,
  title = {mmSampler: Efficient Frame Sampler for Multimodal Video Retrieval},
  author = {Hu, Zhiming and Ye, Ning and Mohomed, Iqbal},
  booktitle = {Proceedings of Machine Learning and Systems},
  pages = {153--171},
  year = {2022}
}

@Article{Luo2021CLIP4Clip,
  author  = {Huaishao Luo and Lei Ji and Ming Zhong and Yang Chen and Wen Lei and Nan Duan and Tianrui Li},
  title   = {{CLIP4Clip}: An Empirical Study of CLIP for End to End Video Clip Retrieval},
  journal = {arXiv preprint arXiv:2104.08860},
  year    = {2021},
}
```
