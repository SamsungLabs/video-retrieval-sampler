import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from modeling import clip
from modeling.clip4lvt import CLIPLVT
from utils.flops_table import feat_dim_dict
from modeling.clip_model import Transformer
from efficientnet_pytorch import EfficientNet

from utils.distributed import AllGather
allgather = AllGather.apply


def init_hidden(batch_size, cell_size, device):
    init_cell = torch.Tensor(batch_size, cell_size).zero_()
    init_cell = init_cell.to(device)
    return init_cell


class PolicySampler(nn.Module):
    def __init__(self, config, clip_state_dict):
        super().__init__()

        self.num_frm = config.num_frm
        self.backbone = config.backbone
        self.no_policy = config.no_policy
        self.no_rnn = config.no_rnn
        self.rnn_type = config.rnn
        self.action_dim = 2
        self.hidden_dim = config.hidden_dim
        self.rescale_size = config.rescale_size
        self.diff = config.diff
        self.concat = config.concat
        self.use_diversity_loss = bool(config.diversity_weight)
        self.diversity_type = config.diversity_type
        self.diversity_backbone = config.diversity_backbone
        self.discard_curr_state = config.discard_curr_state
        self.rank = config.rank
        self.world_size = config.world_size

        if config.backbone == "resnet50":
            self.cnn_model = torchvision.models.resnet50(pretrained=True)
            last_layer = "fc"
        elif config.backbone == "mobilenet_v2":
            self.cnn_model = torchvision.models.mobilenet_v2(pretrained=True)
            last_layer = "classifier"
        elif config.backbone.startswith("efficientnet"):
            self.cnn_model = EfficientNet.from_pretrained(config.backbone)
            last_layer = "_fc"
        elif config.backbone == "raw":
            self.cnn_model = None
            backbone_channel_in_size = config.rescale_size * config.rescale_size
        elif config.backbone == "clip":
            self.cnn_model = None
            backbone_channel_in_size = clip_state_dict["text_projection"].shape[1]
        elif config.backbone == "frozen_clip":
            self.cnn_model = None
            self.frozen_clip, _ = clip.load(config.clip_backbone)
            for _, p in self.frozen_clip.named_parameters():
                p.requires_grad = False
            backbone_channel_in_size = clip_state_dict["text_projection"].shape[1]
        else:
            raise NotImplementedError(f"Unknown backbone: {config.backbone}.")

        if self.cnn_model is not None:
            backbone_channel_in_size = feat_dim_dict[config.backbone]
            setattr(self.cnn_model, last_layer, nn.Sequential()) # remove classification layer

        if self.concat:
            backbone_channel_in_size *= 2

        if  self.rnn_type == "transformer":
            transformer_width = self.hidden_dim
            transformer_heads = transformer_width // 64
            self.projection = self.prepare_linear(backbone_channel_in_size, transformer_width, None, "fc")
            self.frame_position_embeddings = nn.Embedding(config.num_frm, transformer_width)
            self.transformer = Transformer(width=transformer_width, layers=1, heads=transformer_heads)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=backbone_channel_in_size, hidden_size=self.hidden_dim, bias=True, batch_first=True, bidirectional=False)
        elif self.rnn_type == "bilstm":
            self.rnn = nn.LSTM(input_size=backbone_channel_in_size, hidden_size=self.hidden_dim, bias=True, batch_first=True, bidirectional=True)

        input_dim = backbone_channel_in_size if self.no_rnn else self.hidden_dim
        self.linear = self.prepare_linear(input_dim, self.action_dim, config.mlp_hidden_dim, config.mlp_type)

        self.clip = CLIPLVT(clip_state_dict)

        if self.use_diversity_loss:
            if self.diversity_backbone == "clip":
                if hasattr(self, "frozen_clip"):
                    self.diversity_model = self.frozen_clip
                else:
                    self.diversity_model, _ = clip.load(config.clip_backbone)
            else:
                self.diversity_model = getattr(torchvision.models, self.diversity_backbone)(pretrained=True)
                if self.diversity_backbone.startswith("res"):
                    last_layer = "fc"
                elif self.diversity_backbone == "mobilenet_v2":
                    last_layer = "classifier"
                setattr(self.diversity_model, last_layer, nn.Sequential()) # remove classification layer
            for _, p in self.diversity_model.named_parameters():
                p.requires_grad = False

    def prepare_linear(self, input_dim, output_dim, hidden_dim, mlp_type):
        if mlp_type == "fc":
            linear_model = nn.Sequential(nn.Linear(input_dim, output_dim))
        elif mlp_type == "mlp":
            linear_model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            raise NotImplementedError(f"Unknown MLP type: {mlp_type}.")
        for module in linear_model:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
        return linear_model

    def get_policy_actions(self, visual_inputs, tau):

        if self.cnn_model is not None:
            feat_lite = self.cnn_model(visual_inputs) # (b * n_frms, feat_dim)
        elif self.backbone == "clip":
            feat_lite = visual_inputs
        elif self.backbone == "frozen_clip":
            feat_lite = self.frozen_clip.encode_image(visual_inputs).float()
        elif self.backbone == "raw":
            feat_lite = visual_inputs
        feat_lite = feat_lite.view(-1, self.num_frm, feat_lite.size(-1))

        if self.diff or self.concat:
            shifted_feat = torch.roll(feat_lite, 1, 1)
            shifted_feat[:, 0, :] = 0
            if self.diff:
                feat_lite = torch.sub(feat_lite, shifted_feat)
            elif self.concat:
                feat_lite = torch.cat((feat_lite, torch.sub(feat_lite, shifted_feat)), dim=-1)

        if self.no_rnn:
            pass
        elif self.rnn_type == "transformer":
            feat_lite = self.projection(feat_lite)
            position_ids = torch.arange(self.num_frm, dtype=torch.long, device=feat_lite.device)
            position_ids = position_ids.unsqueeze(0).expand(feat_lite.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            feat_lite = feat_lite + frame_position_embeddings
            feat_lite = self.transformer(feat_lite)
        elif self.rnn_type == "lstm":
            feat_lite, _ = self.rnn(feat_lite)
        elif self.rnn_type == "bilstm":
            feat_lite, _ = self.rnn(feat_lite)
            feat_lite = feat_lite.view(-1, self.num_frm, 2, self.hidden_dim)
            feat_lite = torch.mean(feat_lite, dim=2)
        prob = torch.log(F.softmax(self.linear(feat_lite), dim=-1).clamp(min=1e-8))
        actions = F.gumbel_softmax(prob, tau, True)
        actions[:, 0, 0] = 1
        actions[:, 0, 1] = 0
        return actions

    def forward(self, text_input_ids, clip_inputs, policy_inputs, tau=1, return_embds=False):

        r"""Modified from BertModel
        text_inputs: (Bt, num_tokens)
        visual_inputs: (Bv, #frame, C, H, W)
        visual_input_mask: (Bt, #frame)  with 1 indicates valid, 0 indicates invalid position.
        """

        text_embd = self.clip.get_text_output(text_input_ids)
        clip_feats = self.clip.get_visual_output(clip_inputs) # (B, num_frames, feat_dim)

        b, num_frm, c, h, w = policy_inputs.shape
        policy_inputs = policy_inputs.view(b * num_frm, c, h, w)
        if self.backbone == "raw":
            policy_inputs = policy_inputs.view(b * num_frm, -1)
        if self.no_policy:
            actions = torch.zeros(b, num_frm, self.action_dim).to(policy_inputs.device)
            actions[:, :, 0] = 1
        else:
            visual_inputs = clip_feats if self.backbone == "clip" else policy_inputs
            actions = self.get_policy_actions(visual_inputs, tau)  # (B, num_frm, action_size) of 0's and 1's

        visual_input_mask = actions[:, :, 0] # (B, num_frm)

        clip_feats = clip_feats / clip_feats.norm(dim=-1, keepdim=True)
        visual_embd = self.clip._mean_pooling_for_similarity_visual(clip_feats, visual_input_mask)

        if self.use_diversity_loss:
            if self.diversity_backbone == "clip":
                diversity_loss = self.compute_diversity_loss(clip_inputs, visual_input_mask)
            else:
                diversity_loss = self.compute_diversity_loss(policy_inputs, visual_input_mask)
        else:
            diversity_loss = torch.tensor(0.).to(clip_inputs.device)

        if return_embds:
            return {
                'visual_embd': visual_embd,
                'text_embd': text_embd,
                'actions': actions,
                'diversity_loss': diversity_loss
            }
        else:
            visual_embd = allgather(visual_embd, self.rank, self.world_size)
            text_embd = allgather(text_embd, self.rank, self.world_size)
            actions = allgather(actions, self.rank, self.world_size)
            torch.distributed.barrier()
            sim_matrix = self.clip.compute_sim_matrix(
                visual_embd=visual_embd,
                text_embd=text_embd,
            )
            return sim_matrix, actions, diversity_loss

    def compute_diversity_loss(self, visual_inputs, visual_input_mask):
        b, num_frm = visual_inputs.shape[:2]
        diversity_loss = 0.
        if self.diversity_type == "features": # CLIP features
            if self.diversity_backbone == "clip":
                visual_feats = self.diversity_model.encode_image(visual_inputs).float()
            else:
                visual_feats = self.diversity_model(visual_inputs)
            visual_feats = visual_feats.view(b, num_frm, visual_feats.size(-1))
        else:
            visual_feats = visual_inputs.view(b, num_frm, -1)
        visual_feats = visual_feats / (visual_feats.norm(dim=-1, keepdim=True) + 1e-9)

        mask_matrix = torch.bmm(visual_input_mask.unsqueeze(-1), visual_input_mask.unsqueeze(-2))
        diversity_sim_matrix = torch.bmm(visual_feats, visual_feats.permute(0,2,1)) if self.diversity_type == "features" \
                                else torch.cdist(visual_feats, visual_feats, p=2)
        sim_matrix_filtered = diversity_sim_matrix * mask_matrix

        acc_batch_size = 0
        for batch_idx in range(b):
            batch_mask_matirx = mask_matrix[batch_idx, :, :]
            num_frm_keep = visual_input_mask[batch_idx].sum()
            batch_sim_matrix_filtered = sim_matrix_filtered[batch_idx, :, :]
            if num_frm_keep == 1:
                batch_diversity_loss = 0 if self.diversity_type == "features" else 1
            else:
                batch_diversity_loss = (batch_sim_matrix_filtered.sum() - batch_sim_matrix_filtered.trace()) / (num_frm_keep*(num_frm_keep-1))
                acc_batch_size += 1
            diversity_loss += batch_diversity_loss
        diversity_loss = diversity_loss / acc_batch_size if acc_batch_size > 0 else torch.tensor(0.).to(visual_inputs.device)
        return diversity_loss

    def freeze_cnn_backbone(self):
        if self.cnn_model is not None:
            for _, p in self.cnn_model.named_parameters():
                p.requires_grad = False

    def freeze_clip(self):
        for _, p in self.clip.named_parameters():
            p.requires_grad = False
