import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from fedformer.model import Model
from scripts.data_utils import load_data
from scripts.eval_utils import evaluate_metrics
import yaml
import argparse

# Initialize DDP
def setup():
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    torch.distributed.destroy_process_group()

# Config wrapper class
class ConfigWrapper:
    def __init__(self, d):
        self.__dict__.update(d)

def main():
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # Load config
    with open('configs/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    train_loader, test_loader = load_data(
        'data/processed_solexs.csv',
        seq_len=cfg['seq_len'],
        label_len=cfg['label_len'],
        pred_len=cfg['pred_len'],
        distributed=True  # Custom flag to enable DistributedSampler
    )

    model_cfg = {
        **cfg,
        "version": "Fourier",
        "mode_select": "random",
        "modes": 32,
        "moving_avg": 25,
        "L": 1,
        "base": "legendre",
        "cross_activation": "tanh",
        "freq": "h",
        "embed": "timeF",
        "output_attention": False,
        "n_heads": 8,
        "activation": "gelu",
        "wavelet": 0,
        "d_ff": 256
    }

    model = Model(ConfigWrapper(model_cfg)).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=float(cfg['learning_rate']))

    for epoch in range(cfg['epochs']):
        model.train()
        total_loss = 0

        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        for seq_x, seq_y, target in train_loader:
            seq_x = seq_x.to(device)
            seq_y = seq_y.to(device)
            target = target.to(device)

            B, L, _ = seq_x.shape
            mark_enc = torch.zeros((B, L, 4), device=device)
            mark_dec = torch.zeros((B, seq_y.shape[1], 4), device=device)

            optimizer.zero_grad()
            out = model(seq_x, mark_enc, seq_y, mark_dec)
            out = out[:, -1, 0]
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if local_rank == 0:
            print(f"[GPU {local_rank}] Epoch {epoch} - Train Loss: {total_loss / len(train_loader):.4f}")
            if (epoch + 1) % 5 == 0:
                evaluate_metrics(model.module, test_loader, device)

    cleanup()

if __name__ == "__main__":
    main()
