import sys
import json
import argparse

from easydict import EasyDict as edict

def parse_with_config(parsed_args):
    """This function will set args based on the input config file.
        it only overwrites unset parameters,
        i.e., these parameters not set from user command line input
    """
    # convert to EasyDict object, enabling access from attributes even for nested config
    # e.g., args.train_datasets[0].name
    args = edict(vars(parsed_args))
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split("=")[0] for arg in sys.argv[1:]
                         if arg.startswith("--")}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    del args.config
    return args

parser = argparse.ArgumentParser(description="PyTorch implementation of Transformer Video Retrieval")

parser.add_argument('--train_annot', default="", help='json file containing training video annotations')
parser.add_argument('--val_annot', default="", help='json file containing validation video annotations')
parser.add_argument('--test_annot', default="", help='json file containing test video annotations')
parser.add_argument('--clip_path', default="", help='path to finetuned CLIP')
parser.add_argument('--policy_path', default="", help='path to finetuned policy backbone weights')
parser.add_argument('--frames_dir', default="", type=str, help='path to video frames')
parser.add_argument("--output_dir", type=str, default="output", help="dir to store model checkpoints & training meta.")
parser.add_argument("--tensorboard_dir", type=str, default="tensorboard", help="dir to store tensorboard")
parser.add_argument('--config', help='config file path')

# ========================= Model Configs ==========================
parser.add_argument("--backbone", default="mobilenet_v2", type=str, choices=["raw", "clip", "frozen_clip", "resnet50", "mobilenet_v2", "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4", "efficientnet-b5"], help="type of visual backbone")
parser.add_argument("--clip_backbone", default="ViT-B/32", type=str, choices=["ViT-B/32", "ViT-B/16"], help="type of visual backbone for CLIP")
parser.add_argument("--rnn", default="transformer", type=str, choices=["lstm", "bilstm", "transformer"], help="type of RNN backbone")
parser.add_argument("--hidden_dim", default=512, type=int, help="RNN hidden dim")
parser.add_argument("--mlp_hidden_dim", default=1024, type=int, help="MLP hidden dim (used for raw pixel differences)")
parser.add_argument("--mlp_type", type=str, default="mlp", choices=["mlp", "fc"], help="type of linear model to use before gumbel softmax")
parser.add_argument("--rescale_size", default=56, type=int, help="Rescale size for using pixel differences (no CNN backbone)")
parser.add_argument("--no_policy", action="store_true", help="no policy network")
parser.add_argument("--no_rnn", action="store_true", help="no rnn to encode visual features")
parser.add_argument("--diff", action="store_true", help="use feature differences between frames")
parser.add_argument("--concat", action="store_true", help="concat input visual features with difference features for the policy network")
parser.add_argument("--freeze_layer_num", type=int, default=0, help="layer NO. of CLIP need to freeze")

# ========================= Preprocessing Configs ==========================
parser.add_argument("--max_txt_len", type=int, default=20, help="max text #tokens ")
parser.add_argument('--concat_captions', default="concat", choices=["concat", "indep-all", "indep-one"], help='concatenate video captions')
parser.add_argument("--max_img_size", type=int, default=224, help="max image longer side size, shorter side will be padded with zeros")
parser.add_argument("--num_frm", type=int, default=2, help="#frames to use per video")
parser.add_argument("--num_frm_subset", type=int, default=0, help="ablation study: number of frames to sample from num_frm frames")
parser.add_argument("--num_caps", type=int, default=-1, help="number of eval captions per video (if more than 1), should be same for all vids")
parser.add_argument("--sampling", type=str, default="uniform", choices=["uniform", "random"], help="ablation study: how to sample frames")
parser.add_argument("--img_tmpl", default="image_{:05d}.jpg", type=str, help="frame filename pattern")

# ========================= Learning Configs ==========================
parser.add_argument("--train_batch_size", default=32, type=int, help="single-GPU batch size for training.")
parser.add_argument("--val_batch_size", default=32, type=int, help="single-GPU batch size for validation.")
parser.add_argument("--num_epochs", default=30, type=int, help="total # of training epochs.")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="initial learning rate.")
parser.add_argument('--coef_lr', type=float, default=1e-3, help='lr multiplier for clip branch')
parser.add_argument("--no_warmup", action="store_true", help="do not perform cosine warmup LR")
parser.add_argument("--warmup_proportion", default=0.1, type=float, help="proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training")
parser.add_argument("--optim", default="bertadam", choices=["bertadam", "adamw"], help="optimizer")
parser.add_argument("--betas", default=[0.9, 0.98], nargs=2, help="beta for adam optimizer")
parser.add_argument("--weight_decay", default=0.2, type=float, help="weight decay (L2) regularization")
parser.add_argument("--grad_norm", default=1.0, type=float, help="gradient clipping (-1 for no clipping)")
parser.add_argument("--freeze_cnn", action="store_true", help="freeze CNN by setting the requires_grad=False for CNN parameters.")
parser.add_argument("--freeze_clip", action="store_true", help="freeze CLIP by setting the requires_grad=False for CLIP parameters.")
parser.add_argument("--discard_curr_state", action="store_true", help="discard current lstm states if current frame is skipped")
parser.add_argument("--retrieval_weight", default=1, type=float, help="weight factor for retrieval loss")
parser.add_argument("--flop_weight", default=0, type=float, help="weight factor for GFLOPs loss")
parser.add_argument("--uniform_weight", default=0, type=float, help="weight factor for uniform loss")
parser.add_argument("--diversity_weight", default=0, type=float, help="weight factor for visual diversity loss")
parser.add_argument("--diversity_type", default="raw", choices=["features", "raw"], help="use visual features or raw pixels for visual diversity loss")
parser.add_argument("--diversity_backbone", default="clip", choices=["clip", "mobilenet_v2", "resnet152", "resnext101_32x8d"], help="visual backbone for diversity loss")
parser.add_argument('--init_tau', default=5.0, type=float, help="annealing init temperature")
parser.add_argument('--exp_decay_factor', default=-0.045, type=float, help="exp decay factor per epoch")

# ========================= Runtime Configs ==========================
parser.add_argument("--resume", type=str, help="path to latest checkpoint")
parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
parser.add_argument("--num_workers", type=int, default=4, help="#workers for data loading")
parser.add_argument("--do_inference", action="store_true", help="perform inference run")
parser.add_argument("--pin_mem", action="store_true", help="pin memory")
parser.add_argument("--debug", action="store_true", help="debug mode. Log extra information")
parser.add_argument("--data_subset", default=0, type=int, help="debug mode. Use only this number of samples for training and testing")
parser.add_argument("--no_output", action="store_true", help="do not save model and logs")
parser.add_argument("--multi_cap_eval", type=int, default=0, choices=[0, 1], help="whether multiple queries per video are used in evaluation")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                help="# of updates steps to accumulate before performing a backward/update pass."
                "Used to simulate larger batch size training. The simulated batch size "
                "is train_batch_size * gradient_accumulation_steps for a single GPU.")
parser.add_argument('--n_display', type=int, default=20, help='Information display frequency')

# ========================= Distributed Configs ==========================
parser.add_argument("--world_size", default=1, type=int, help="distributed training")
parser.add_argument("--local_rank", default=0, type=int, help="distributed training")
parser.add_argument("--rank", default=0, type=int, help="distributed training")

# ========================= MLFlow Configs ==========================
parser.add_argument("--log_mlflow", action="store_true", help="log params and metrics to MLFlow")
parser.add_argument("--exp_name", default="sampler-activitynet", help="current experiment name")
parser.add_argument("--tracking_uri", default="http://deplo-mlflo-1n72ddjx4k9hz-85103bdf1fd6c6ae.elb.ca-central-1.amazonaws.com/", help="URI to remote server")
