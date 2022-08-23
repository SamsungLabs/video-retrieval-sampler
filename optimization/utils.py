import torch
from optimization.bertadam import BertAdam


def setup_optimizer_and_scheduler(model, cfg, num_train_steps=-1):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_clip_param_tp], 'weight_decay': cfg.weight_decay, 'lr': cfg.learning_rate * cfg.coef_lr},
        {'params': [p for _, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': cfg.learning_rate * cfg.coef_lr},
        {'params': [p for _, p in decay_noclip_param_tp], 'weight_decay': cfg.weight_decay},
        {'params': [p for _, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    t_total = -1 if cfg.no_warmup else num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters, lr=cfg.learning_rate, betas=cfg.betas, weight_decay=cfg.weight_decay, 
                schedule='warmup_cosine', warmup=cfg.warmup_proportion, t_total=t_total)
 
    scheduler = None

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank], output_device=cfg.local_rank, find_unused_parameters=True)

    return model, optimizer, scheduler