import os
import math
import torch
import mlflow
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

from utils.flops_table import get_gflops_params
from utils.logger import LOGGER, add_log_to_file
from torch.utils.tensorboard import SummaryWriter
from utils.basic_utils import set_seeds, save_json, NoOp
from utils.distributed import all_gather, is_main_process, reduce_loss_dict
from utils.train_utils import progress, save_checkpoint, verbose, log_metrics

from modeling.loss import CrossEn
from modeling.clip_model import CLIP
from modeling.model import PolicySampler
from datasets.dataset import BaseDataset
from datasets.prefetch import PrefetchLoader
from configs.config import parser, parse_with_config
from modeling.metrics import t2v_metrics, v2t_metrics
from optimization.utils import setup_optimizer_and_scheduler

torch.distributed.init_process_group(backend="nccl")


def setup_model(cfg, device):
    LOGGER.info("Setup model...")

    pretrained_state_dict = CLIP.get_config(pretrained_clip_name=cfg.clip_backbone)
    state_dict = {}
    epoch = 0
    if cfg.resume:
        LOGGER.info(f"Loading model checkpoint: {cfg.resume}...")
        checkpoint = torch.load(cfg.resume, map_location="cpu")
        state_dict = checkpoint['state_dict']
        epoch = checkpoint["epoch"]
    else:
        if cfg.clip_path:
            LOGGER.info(f"Loading CLIP checkpoint: {cfg.clip_path}...")
            checkpoint = torch.load(cfg.clip_path, map_location="cpu")
            finetuned_state_dict = checkpoint['state_dict']
            for key, val in finetuned_state_dict.items():    
                new_key = "clip." + key
                if new_key not in state_dict:
                    state_dict[new_key] = val.clone()
        else:
            LOGGER.info(f"Using CLIP pretrained weights...")
            for key, val in pretrained_state_dict.items():    
                new_key = "clip.clip." + key
                if new_key not in state_dict:
                    state_dict[new_key] = val.clone()
        if cfg.policy_path:
            LOGGER.info(f"Loading policy checkpoint: {cfg.policy_path}...")
            checkpoint = torch.load(cfg.policy_path, map_location="cpu")
            finetuned_state_dict = checkpoint['state_dict']
            for key, val in finetuned_state_dict.items():    
                if key.startswith("cnn_model"):
                    state_dict[key] = val.clone()

    model = PolicySampler(cfg, pretrained_state_dict)
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    if cfg.debug:
        LOGGER.info("-" * 20)
        if len(missing_keys) > 0:
            LOGGER.info("Weights of {} not initialized from pretrained model: {}"
                        .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
        if len(unexpected_keys) > 0:
            LOGGER.info("Weights from pretrained model not used in {}: {}"
                        .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
        if len(error_msgs) > 0:
            LOGGER.error("Weights from pretrained model cause errors in {}: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

    if str(device) == "cpu":
        model.float()

    if cfg.freeze_clip:
        model.freeze_clip()
    if cfg.freeze_cnn:
        model.freeze_cnn_backbone()

    model.to(device)

    LOGGER.info("Setup model done!")
    return model, epoch


def setup_dataloaders(cfg, device, train_annot, val_annot):

    LOGGER.info("Init. train_loader and val_loader...")

    train_dataset = BaseDataset(cfg, train_annot, is_train=True)
    val_dataset = BaseDataset(cfg, val_annot, is_train=False)

    sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train_batch_size,
        num_workers=cfg.num_workers,
        collate_fn=train_dataset.collate_data,
        pin_memory=cfg.pin_mem,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.num_workers,
        collate_fn=val_dataset.collate_data,
        pin_memory=cfg.pin_mem,
        shuffle=False,
        drop_last=False,
    )

    if str(device) != "cpu":
        train_loader = PrefetchLoader(train_loader)
        val_loader = PrefetchLoader(val_loader)
    
    LOGGER.info("Init. train_loader and val_loader done!")

    return train_loader, sampler, val_loader


def init_gflops_table(cfg):
    gflops_table = {}

    gflops_table["clip"] = get_gflops_params("CLIP", cfg)
    gflops_table["policy"] = get_gflops_params(cfg.backbone, cfg)
    gflops_table[cfg.rnn] = 0 if cfg.no_rnn else get_gflops_params(cfg.rnn, cfg)
    gflops_table["mlp"] = get_gflops_params("mlp", cfg)

    LOGGER.info("gflops_table: ")
    for k in gflops_table:
        LOGGER.info("%-20s: %.4f GFLOPS" % (k, gflops_table[k]))

    return gflops_table


def get_gflops_and_frames(gflops_table, clip_backbone=False):
    gflops_vec = [gflops_table["clip"]]
    frames_vec = [1]
    if clip_backbone:
        gflops_vec.append(gflops_table["clip"])
    else:
        gflops_vec.append(0)
    frames_vec.append(0)
    return gflops_vec, frames_vec


def compute_flop_loss(actions, gflops_table, clip_backbone):
    gflops, _ = get_gflops_and_frames(gflops_table, clip_backbone)
    gflops = torch.tensor(gflops).to(actions.device)
    flop_loss = torch.sum(torch.mean(actions, dim=[0, 1]) * gflops)

    policy_dim = 2
    action_distribution = torch.zeros(policy_dim).to(actions.device)
    for i in range(policy_dim):
        action_distribution[i] = torch.sum(actions[:, :, i])
    action_distribution = action_distribution / torch.sum(action_distribution)
    usage_bias = action_distribution - torch.mean(action_distribution)
    uniform_loss = torch.norm(usage_bias, p=2)

    return flop_loss, uniform_loss


def log_policy_usage(actions, gflops_table, cfg):
    gflops, frames = get_gflops_and_frames(gflops_table, clip_backbone=cfg.backbone=="clip")

    tmp_cnt = [np.sum(actions[:, :, iii] == 1) for iii in range(actions.shape[2])] # Get count for each action
    tmp_total_cnt = sum(tmp_cnt)

    avg_gflops = 0
    avg_frame_ratio = 0

    for i in range(actions.shape[2]):
        usage_ratio = tmp_cnt[i] / tmp_total_cnt
        if i == 0:
            LOGGER.info(f"CLIP model: {tmp_cnt[i]} ({100 * usage_ratio:.2f})%")
        else:
            LOGGER.info(f"Skip 1 frame: {tmp_cnt[i]} ({100 * usage_ratio:.2f})%")

        avg_gflops += usage_ratio * gflops[i]
        avg_frame_ratio += usage_ratio * frames[i]

    if not cfg.no_policy:
        avg_gflops += (gflops_table["policy"] + gflops_table[cfg.rnn] + gflops_table["mlp"])

    num_frm = cfg.num_frm if cfg.num_frm_subset <= 0 else min(cfg.num_frm, cfg.num_frm_subset)
    LOGGER.info(f"GFLOPS/f: {avg_gflops:.3f} GFLOPS/v: {avg_gflops*cfg.num_frm:.3f} AVG_FRAMES: {avg_frame_ratio * num_frm:.3f}")
    return avg_frame_ratio * num_frm


def get_current_temperature(cfg, epoch):
    return cfg.init_tau * np.exp(cfg.exp_decay_factor * epoch)


def get_embeddings(val_loader, model, tau, cfg):
    with torch.no_grad():
        visual_embd = []
        text_embd = []
        actions = []
        diversity_loss = []
        break_pts = [0]
        if is_main_process():
            pbar = tqdm(total=len(val_loader), desc="Evaluation", unit="batch")
        else:
            pbar = NoOp()

        for minibatch in val_loader:
            output = model(**minibatch, tau=tau, return_embds=True)
            visual_embd.append(output["visual_embd"]) # (B, feat_dim)
            if cfg.multi_cap_eval and cfg.num_caps == -1: # MSVD, multi-caption eval, output["text_embd"] = (B=1, num_caps, feat_dim)
                output["text_embd"] = output["text_embd"].squeeze(0).unsqueeze(1) # (num_caps, 1, feat_dim)
                break_pts.append(break_pts[-1] + output["text_embd"].shape[0])
            elif cfg.num_caps != -1:
                b, num_caps, _ = output["text_embd"].shape
                assert cfg.num_caps == num_caps
                output["text_embd"] = output["text_embd"].view(b*num_caps, -1).unsqueeze(1)
            text_embd.append(output["text_embd"])
            actions.append(output["actions"]) # (B, num_frm, num_actions)
            diversity_loss.append(output["diversity_loss"])

            pbar.update(1)
        pbar.close()

        visual_embd = torch.cat(visual_embd, 0)
        text_embd = torch.cat(text_embd, 0)
        actions = torch.cat(actions, 0)
        diversity_loss = torch.mean(torch.stack(diversity_loss))

        if cfg.num_caps != -1:
            num_vids = visual_embd.shape[0]
            break_pts = list(range(0, (num_vids+1)*cfg.num_caps, cfg.num_caps))
        if break_pts == [0]:
            break_pts = None

        res = {
            "visual_embd": visual_embd,
            "text_embd": text_embd,
            "actions": actions,
        }

        return res, break_pts, diversity_loss


def reshape_sim_matrix(sims, break_pts):
    num_t, num_v = sims.shape
    if num_t == num_v:
        return sims
    sims_reshaped = torch.zeros((num_v, num_v)).to(sims.device)
    for v in range(num_v):
        for i in range(len(break_pts)-1):
            sims_reshaped[i, v] = torch.max(sims[break_pts[i]:break_pts[i+1], v], dim=0)[0]
    return sims_reshaped


@torch.no_grad()
def validate(model, val_loader, device, cfg, tau=0.001, criterion=None, writer=None, epoch=None, gflops_table=None):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
    model.eval()

    embds, break_pts, diversity_loss = get_embeddings(val_loader, model, tau, cfg)

    visual_embd = embds["visual_embd"]
    text_embd = embds["text_embd"]
    actions = embds["actions"]

    sims = model.clip.compute_sim_matrix(
        visual_embd=visual_embd,
        text_embd=text_embd,
    )

    LOGGER.info(f"Num. of queries: {sims.shape[0]}, Num. of videos: {sims.shape[1]}")

    tv_metrics = t2v_metrics(sims, break_pts)
    vt_metrics = v2t_metrics(sims, break_pts)
    all_metrics = {"t2v_metrics": tv_metrics, "v2t_metrics": vt_metrics}

    if is_main_process() and criterion:
        reshaped_sims = reshape_sim_matrix(sims, break_pts)
        loss1 = criterion(reshaped_sims)
        loss2 = criterion(reshaped_sims.T)
        retrieval_loss = (loss1 + loss2) / 2
        flop_loss, uniform_loss = compute_flop_loss(actions, gflops_table, clip_backbone=cfg.backbone=="clip")
        writer.add_scalar('Retrieval Loss/val', retrieval_loss.item(), epoch)
        writer.add_scalar('FLOP Loss/val', flop_loss.item(), epoch)
        writer.add_scalar('Uniform Loss/val', uniform_loss.item(), epoch)
        writer.add_scalar('Diversity Loss/val', diversity_loss.item(), epoch)
        loss = cfg.retrieval_weight * retrieval_loss + cfg.flop_weight * flop_loss + cfg.uniform_weight * uniform_loss + cfg.diversity_weight * diversity_loss
        writer.add_scalar('Total Epoch Loss/val', loss.item(), epoch)
        LOGGER.info(f"EVAL epoch {epoch} Loss: {(loss.item()):.6f}")
        LOGGER.info(f"Retrieval Loss: {retrieval_loss.item():.3f}, FLOP Loss: {flop_loss.item():.3f}, Uniform Loss: {uniform_loss.item():.3f}, Diversity Loss: {diversity_loss.item():.3f}")
        if cfg.log_mlflow:
            mlflow.log_metric('Retrieval Loss/val', retrieval_loss.item())
            mlflow.log_metric('FLOP Loss/val', flop_loss.item())
            mlflow.log_metric('Uniform Loss/val', uniform_loss.item())
            mlflow.log_metric('Diversity Loss/val', diversity_loss.item())
            mlflow.log_metric('Total Epoch Loss/val', loss.item())
    actions = actions.cpu().detach().numpy()
    avg_frames = log_policy_usage(actions, gflops_table, cfg)
    if cfg.do_inference and cfg.log_mlflow:
        mlflow.log_metric('Average Frames/test', avg_frames)

    return all_metrics, actions


def train(cfg):

    if is_main_process() and cfg.log_mlflow:
        mlflow.set_tracking_uri(cfg.tracking_uri)
        mlflow.set_experiment(cfg.exp_name)

    set_seeds(cfg.seed)

    if not cfg.train_annot or not cfg.val_annot:
        raise ValueError("Empty annotation path!")

    if cfg.multi_cap_eval and cfg.num_caps == -1:
        cfg.val_batch_size = 1 # Force batch size to be 1 for MSVD

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(cfg.local_rank)
    cfg.world_size = world_size
    rank = torch.distributed.get_rank()
    cfg.rank = rank
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", cfg.local_rank)

    if is_main_process():
        writer = SummaryWriter(cfg.tensorboard_dir)
    else:
        LOGGER.disabled = True
        writer = NoOp()

    if not cfg.no_output:
        timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(cfg.output_dir, "log", timestamp)
        add_log_to_file(os.path.join(log_dir, "log.info"))
        
        ckpt_dir = os.path.join(cfg.output_dir, "models", timestamp)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        save_json(cfg, os.path.join(ckpt_dir, 'config.json'), save_pretty=True)

    if is_main_process() and cfg.log_mlflow:
        mlflow.log_params(cfg)

    model, _ = setup_model(cfg, device=device)

    train_loader, sampler, val_loader = setup_dataloaders(cfg, device, cfg.train_annot, cfg.val_annot)

    total_train_batch_size = int(cfg.world_size * cfg.train_batch_size * cfg.gradient_accumulation_steps)
    num_train_steps = int(math.ceil(1. * cfg.num_epochs * len(train_loader.dataset) / total_train_batch_size))

    LOGGER.info(f"device: {device} n_gpu: {cfg.world_size}, "
                f"rank: {cfg.rank}")
    LOGGER.info("Starting training...")
    LOGGER.info(f"***** Running training on {cfg.world_size} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_loader.dataset))
    LOGGER.info("  Batch size = %d", cfg.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", cfg.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", num_train_steps)
    LOGGER.info(f"***** Validation information *****")
    LOGGER.info("  Num examples = %d", len(val_loader.dataset))
    LOGGER.info("  Batch size = %d", cfg.val_batch_size)
    LOGGER.info("  Num steps = %d", len(val_loader))

    assert cfg.freeze_layer_num <= 12 and cfg.freeze_layer_num >= -1
    if hasattr(model, "clip") and cfg.freeze_layer_num > -1 and not cfg.freeze_clip:
        for name, param in model.clip.clip.named_parameters():
            # top layers always need to train
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                continue    # need to train
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= cfg.freeze_layer_num:
                    continue    # need to train
            # paramenters which < freeze_layer_num will be freezed
            param.requires_grad = False

    gflops_table = init_gflops_table(cfg)

    criterion = CrossEn()

    model, optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, cfg, num_train_steps)

    best = -np.inf
    best_metrics, best_actions = None, None
    global_step = 0
    len_epoch = len(train_loader)
    for epoch in range(cfg.num_epochs):

        set_seeds(cfg.seed + epoch)

        total_loss = 0
        model.train()

        sampler.set_epoch(epoch)
        all_actions_list = []
        tau = get_current_temperature(cfg, epoch)

        for step, minibatch in enumerate(train_loader):

            sim_matrix, actions, diversity_loss = model(**minibatch, tau=tau)
            all_actions_list.append(actions.cpu().detach().numpy())
            loss1 = criterion(sim_matrix)
            loss2 = criterion(sim_matrix.T)
            retrieval_loss = (loss1 + loss2) / 2

            flop_loss, uniform_loss = compute_flop_loss(actions, gflops_table, clip_backbone=cfg.backbone=="clip")
            loss = cfg.retrieval_weight * retrieval_loss + cfg.flop_weight * flop_loss + cfg.uniform_weight * uniform_loss + cfg.diversity_weight * diversity_loss

            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps
            loss.backward()

            # Reduce losses over all GPUs for logging purposes
            loss_dict = {"Retrieval Loss": retrieval_loss, "FLOP Loss": flop_loss, "Uniform Loss": uniform_loss, "Diversity Loss": diversity_loss}
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = cfg.retrieval_weight * loss_dict_reduced["Retrieval Loss"] + cfg.flop_weight * loss_dict_reduced["FLOP Loss"] + cfg.uniform_weight * loss_dict_reduced["Uniform Loss"] + cfg.diversity_weight * loss_dict_reduced["Diversity Loss"]
            total_loss += losses_reduced.item()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:

                if cfg.grad_norm != -1:
                    clip_grad_norm_(model.parameters(), cfg.grad_norm)

                if lr_scheduler is not None:
                    lr_scheduler.step()

                optimizer.step()
                optimizer.zero_grad()

                # https://github.com/openai/CLIP/issues/46
                if hasattr(model, 'module'):
                    torch.clamp_(model.module.clip.clip.logit_scale.data, max=np.log(100))
                else:
                    torch.clamp_(model.clip.clip.logit_scale.data, max=np.log(100))

                global_step += 1
                if global_step % cfg.n_display == 0:
                    prog = progress(step+1, len_epoch)
                    LOGGER.info(f"Train Epoch: {epoch} {prog} Loss: {losses_reduced.item():.6f} Lr: {'-'.join([str('%.9f'%itm) for itm in sorted(list(set(optimizer.get_lr())))])}")
                    LOGGER.info("  ".join([f"{k}: {v.item():.3f}" for k, v in loss_dict_reduced.items()]))
                    log_policy_usage(actions.cpu().detach().numpy(), gflops_table, cfg)
                writer.add_scalar('Retrieval Loss/train', loss_dict_reduced["Retrieval Loss"].item(), global_step)
                writer.add_scalar('FLOP Loss/train', loss_dict_reduced["FLOP Loss"].item(), global_step)
                writer.add_scalar('Uniform Loss/train', loss_dict_reduced["Uniform Loss"].item(), global_step)
                writer.add_scalar('Diversity Loss/train', loss_dict_reduced["Diversity Loss"].item(), global_step)
                writer.add_scalar('Total Loss/train', losses_reduced.item(), global_step)

                if is_main_process() and cfg.log_mlflow:
                    mlflow.log_metric('Retrieval Loss/train', loss_dict_reduced["Retrieval Loss"].item())
                    mlflow.log_metric('FLOP Loss/train', loss_dict_reduced["FLOP Loss"].item())
                    mlflow.log_metric('Uniform Loss/train', loss_dict_reduced["Uniform Loss"].item())
                    mlflow.log_metric('Diversity Loss/train', loss_dict_reduced["Diversity Loss"].item())
                    mlflow.log_metric('Total Loss/train', losses_reduced.item())

        LOGGER.info(f"Train Epoch: {epoch} Loss: {(total_loss / len_epoch):.6f}")
        writer.add_scalar('Total Epoch Loss/train', total_loss / len_epoch, epoch)
        if cfg.log_mlflow:
            mlflow.log_metric('Total Epoch Loss/train', total_loss / len_epoch)
        log_policy_usage(np.concatenate(all_actions_list, axis=0), gflops_table, cfg)

        set_seeds(cfg.seed)
        if is_main_process():
            ret_metrics, val_actions = validate(model, val_loader, device, cfg, tau, criterion, writer, epoch, gflops_table)
            for metric in ret_metrics:
                verbose(ret_metrics[metric], metric, epoch)
                log_metrics(ret_metrics[metric], metric, epoch, writer)

            best_recall = ret_metrics["t2v_metrics"]["R1"]
            improved = best_recall > best

            if improved:
                best = best_recall
                best_metrics = ret_metrics
                best_actions = val_actions
                best_checkpoint = {"epoch": epoch, "model": model}
                if not cfg.no_output:
                    save_checkpoint(best_checkpoint, cfg, optimizer, os.path.join(ckpt_dir, "trained_model.pth"))
                    LOGGER.info(f"Saving the best ckpt to disk (epoch {best_checkpoint['epoch']})")
            else:
                LOGGER.info(f"This epoch did not improve R1-5-10. Best checkpoint saved for epoch {best_checkpoint['epoch']}")

    if is_main_process():
        writer.close()
        LOGGER.info(f"Best retrieval performance from epoch {best_checkpoint['epoch']}")
        avg_frames = log_policy_usage(best_actions, gflops_table, cfg)
        for metric in best_metrics:
            verbose(best_metrics[metric], metric, best_checkpoint['epoch'])
        if cfg.log_mlflow:
            for metric in best_metrics:
                for metric_name in best_metrics[metric]:
                    mlflow.log_metric(metric + "/" + metric_name + "/" + "val", best_metrics[metric][metric_name])
            mlflow.log_metric('Average Frames/val', avg_frames)

    if not cfg.no_output:
        log_dirs = all_gather(log_dir)
        ckpt_dirs = all_gather(ckpt_dir)
        if is_main_process():
            print(f"Log file stored at {log_dirs[0]}")
            print(f"The best performing ckpt can be found at {str(ckpt_dirs[0])}")
            if cfg.log_mlflow:
                mlflow.log_param("best_model_path", os.path.join(ckpt_dirs[0], "trained_model.pth"))

        cfg.resume = os.path.join(ckpt_dirs[0], "trained_model.pth")
        cfg.do_inference = True
        if is_main_process():
            test(cfg)


def test(cfg):

    set_seeds(cfg.seed)

    if not cfg.test_annot:
        if not cfg.val_annot:
            raise ValueError("Empty annotation path!")
        cfg.test_annot = cfg.val_annot

    if cfg.multi_cap_eval and cfg.num_caps == -1:
        cfg.val_batch_size = 1 # Force batch size to be 1 for MSVD

    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(cfg.local_rank)
    cfg.world_size = world_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", cfg.local_rank)

    if cfg.local_rank != 0:
        LOGGER.disabled = True

    if not cfg.no_output:
        timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(cfg.output_dir, "log", timestamp)
        add_log_to_file(os.path.join(log_dir, "log.info"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", cfg.local_rank)

    model, epoch = setup_model(cfg, device)

    _, _, eval_loader = setup_dataloaders(cfg, device, cfg.train_annot, cfg.test_annot)

    LOGGER.info(f"***** Validation information *****")
    LOGGER.info("  Num examples = %d", len(eval_loader.dataset))
    LOGGER.info("  Batch size = %d", cfg.val_batch_size)
    LOGGER.info("  Num steps = %d", len(eval_loader))

    gflops_table = init_gflops_table(cfg)

    tau = get_current_temperature(cfg, epoch=epoch)

    if is_main_process():
        ret_metrics, _ = validate(model, eval_loader, device, cfg, tau, gflops_table=gflops_table)
        for metric in ret_metrics:
            verbose(ret_metrics[metric], metric)
        if cfg.log_mlflow:
            for metric in ret_metrics:
                for metric_name in ret_metrics[metric]:
                    mlflow.log_metric(metric + "/" + metric_name + "/" + "test", ret_metrics[metric][metric_name])
    
        if not cfg.no_output:
            save_json(cfg, os.path.join(log_dir, "config.json"), save_pretty=True)
            print(f"Log file stored at {log_dir}")


if __name__ == '__main__':
    parsed_args = parser.parse_args()
    args = parse_with_config(parsed_args)
    if args.do_inference:
        test(args)
    else:
        train(args)
