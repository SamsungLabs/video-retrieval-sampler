import torch
import torch.nn as nn
from modeling.clip_model import build_model


class CLIPLVT(nn.Module):
    def __init__(self, clip_state_dict):
        super(CLIPLVT, self).__init__()

        self.clip = build_model(clip_state_dict)

    def get_text_output(self, text_input_ids):

        b, num_caps, ctx_len = text_input_ids.shape
        text_input_ids = text_input_ids.view(b * num_caps, ctx_len)
        sequence_hidden = self.clip.encode_text(text_input_ids).float()
        sequence_hidden = sequence_hidden.view(b, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, visual_inputs):

        b, num_frms, channel, h, w = visual_inputs.shape
        visual_inputs = visual_inputs.view(b * num_frms, channel, h, w)
        visual_hidden = self.clip.encode_image(visual_inputs).float()
        visual_hidden = visual_hidden.view(b, -1, visual_hidden.size(-1))
        
        return visual_hidden

    def forward(self, text_input_ids, visual_inputs, visual_input_mask=None, flops=False, return_embds=False):

        if flops:
            return self.get_visual_output(visual_inputs)

        if visual_input_mask is None:
            visual_input_mask = torch.ones((visual_inputs.shape[0], visual_inputs.shape[1])).to(visual_inputs.device)

        text_embd = self.get_text_output(text_input_ids)
        visual_embd = self.get_visual_output(visual_inputs)
        visual_embd = self._mean_pooling_for_similarity_visual(visual_embd, visual_input_mask)

        if return_embds:
            return {
                'visual_embd': visual_embd,
                'text_embd': text_embd,
            }
        else:  
            sim_matrix = self.compute_sim_matrix(
                visual_embd=visual_embd,
                text_embd=text_embd,
            )
            return sim_matrix

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def compute_sim_matrix(self, visual_embd, text_embd):

        b, num_caps, _ = text_embd.shape
        text_embd = text_embd.view(b * num_caps, -1)
        text_embd = text_embd / text_embd.norm(dim=-1, keepdim=True)

        visual_embd = visual_embd / visual_embd.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        sims = logit_scale * torch.matmul(text_embd, visual_embd.t())

        return sims
