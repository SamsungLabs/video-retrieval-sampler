import sys
sys.path.insert(0, "../")

import torch
import torchvision
from torch import nn
from modeling.clip import tokenize
from modeling.clip_model import CLIP
from modeling.clip4lvt import CLIPLVT
from modeling.clip_model import Transformer
from ptflops import get_model_complexity_info


feat_dim_dict = {
    "clip": 512,
    "frozen_clip": 512,
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
    "mobilenet_v2": 1280,
    "efficientnet-b0": 1280,
    "efficientnet-b1": 1280,
    "efficientnet-b2": 1408,
    "efficientnet-b3": 1536,
    "efficientnet-b4": 1792,
    "efficientnet-b5": 2048,
    "resnext101_32x8d": 2048,
}

efficientnet_prior_dict = {
    "efficientnet-b0": (0.39, 5.3),
    "efficientnet-b1": (0.70, 7.8),
    "efficientnet-b2": (1.00, 9.2),
    "efficientnet-b3": (1.80, 12),
    "efficientnet-b4": (4.20, 19),
    "efficientnet-b5": (9.90, 30),
}


def get_gflops_params(backbone, cfg):

    macs = 0

    if backbone == "raw" or backbone == "clip":
        pass

    elif backbone.startswith("efficientnet"):
        gflops, _ = efficientnet_prior_dict[backbone]
        return gflops
    
    elif backbone == "transformer":
        transformer_width = cfg.hidden_dim
        transformer_heads = transformer_width // 64
        backbone_channel_in_size = cfg.rescale_size * cfg.rescale_size if cfg.backbone == "raw" else feat_dim_dict[cfg.backbone]
        input_dim = backbone_channel_in_size
        if cfg.concat:
            input_dim *= 2
        model = nn.Sequential(nn.Linear(input_dim, transformer_width),
                            Transformer(width=transformer_width, layers=1, heads=transformer_heads)
        )
        macs, _ = get_model_complexity_info(model, (cfg.num_frm,input_dim), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
        macs /= cfg.num_frm

    elif backbone not in ["CLIP", "frozen_clip", "lstm", "bilstm", "mlp"]:
        if backbone == "resnet50":
            model = torchvision.models.resnet50(pretrained=True)
            last_layer = "fc"
        elif backbone == "mobilenet_v2":
            model = torchvision.models.mobilenet_v2(pretrained=True)
            last_layer = "classifier"
        setattr(model, last_layer, nn.Sequential())

        macs, _ = get_model_complexity_info(model, (3,cfg.max_img_size,cfg.max_img_size), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)

    elif backbone == "CLIP" or backbone == "frozen_clip":
        clip_state_dict = CLIP.get_config(pretrained_clip_name=cfg.clip_backbone)
        model = CLIPLVT(clip_state_dict).float()

        def prepare_inputs(res):
            text_input_ids = torch.tensor(tokenize(["test"])).unsqueeze(0)
            visual_inputs = torch.rand((1, 1, 3, cfg.max_img_size, cfg.max_img_size)).float()
            return dict(text_input_ids=text_input_ids, visual_inputs=visual_inputs, visual_input_mask=None, flops=True)

        macs, _ = get_model_complexity_info(model, (1,), input_constructor=prepare_inputs, as_strings=False,
                                           print_per_layer_stat=False, verbose=False)

    elif backbone == "lstm" or backbone == "bilstm":
        if cfg.backbone == "raw":
            policy_feat_dim = cfg.rescale_size * cfg.rescale_size
        else:
            policy_feat_dim = feat_dim_dict[cfg.backbone]
        if cfg.concat:
            policy_feat_dim *= 2
        bidirectional = backbone == "bilstm"
        model = nn.Sequential(nn.LSTM(input_size=policy_feat_dim, hidden_size=cfg.hidden_dim, bias=True, batch_first=True, bidirectional=bidirectional))

        macs, _ = get_model_complexity_info(model, (cfg.num_frm,policy_feat_dim), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
        macs /= cfg.num_frm

    elif backbone == "mlp":

        backbone_channel_in_size = cfg.rescale_size * cfg.rescale_size if cfg.backbone == "raw" else feat_dim_dict[cfg.backbone]
        input_dim = backbone_channel_in_size if cfg.no_rnn else cfg.hidden_dim
        hidden_dim = cfg.mlp_hidden_dim
        output_dim = 2
        if cfg.mlp_type == "fc":
            linear_model = nn.Linear(input_dim, output_dim)
        elif cfg.mlp_type == "mlp":
            linear_model = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )

        macs, _ = get_model_complexity_info(linear_model, (input_dim,), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)

    gflops = macs / 1e9

    return gflops


if __name__ == "__main__":
    from types import SimpleNamespace
    cfg = {
        "max_img_size": 224, 
        "backbone": "mobilenet_v2", 
        "hidden_dim": 512, 
        "rnn": "lstm", 
        "clip_backbone": "ViT-B/32",
        "concat": False,
        "no_rnn": False,
        "num_frm": 16,
    }
    cfg = SimpleNamespace(**cfg)
    for model in ["resnet50", "mobilenet_v2", "efficientnet-b3", "CLIP", cfg.rnn, "transformer"]:
        gflops = get_gflops_params(model, cfg)
        print("%-20s: %.4f GFLOPS" % (model, gflops))
