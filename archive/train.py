import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from fedformer.model import Model
from scripts.data_utils import load_data
from scripts.eval_utils import evaluate_metrics

with open('configs/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if cfg['use_gpu'] and torch.cuda.is_available() else 'cpu')

train_loader, test_loader = load_data(
    'data/processed_solexs.csv',
    seq_len=cfg['seq_len'],
    label_len=cfg['label_len'],
    pred_len=cfg['pred_len']
)

class ConfigWrapper:
    def __init__(self, d):
        self.__dict__.update(d)

model_cfg = {
    **cfg,
    "version": "Fourier",           # or 'Wavelets'
    "mode_select": "random",
    "modes": 32,
    "moving_avg": 25,
    "L": 1,
    "base": "legendre",
    "cross_activation": "tanh",
    "freq": "h",                    # doesn't matter for dummy marks
    "embed": "timeF",
    "output_attention": False,
    "n_heads": 8,
    "activation": "gelu",
    "wavelet": 0,
    "d_ff": 256
}

model = Model(ConfigWrapper(model_cfg)).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=cfg['learning_rate'])

for epoch in range(cfg['epochs']):
    model.train()
    total_loss = 0

    for seq_x, seq_y, target in train_loader:
        seq_x = seq_x.to(device)
        seq_y = seq_y.to(device)
        target = target.to(device)

        # Dummy time encodings (same shape as x, but 4 features)
        B, L, _ = seq_x.shape
        mark_enc = torch.zeros((B, L, 4)).to(device)
        mark_dec = torch.zeros((B, seq_y.shape[1], 4)).to(device)

        optimizer.zero_grad()
        out = model(seq_x, mark_enc, seq_y, mark_dec)
        out = out[:, -1, 0]  # get final prediction step
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Train Loss: {total_loss / len(train_loader):.4f}")

    if (epoch + 1) % 5 == 0:
        evaluate_metrics(model, test_loader, device)