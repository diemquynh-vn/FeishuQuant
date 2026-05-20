import os
import torch

DATA_ROOT = r'D:\Feishu\data'

LOB_FILE = os.path.join(
    DATA_ROOT,
    'lob_data_in_sample.parquet'
)

DAILY_FILE = os.path.join(
    DATA_ROOT,
    'daily_data_in_sample.parquet'
)

OUT_DIR = os.path.join(
    DATA_ROOT,
    'outputs'
)

os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DAYS = [
    f'D{i:03d}'
    for i in range(1, 388)
]

VAL_DAYS = [
    f'D{i:03d}'
    for i in range(388, 484)
]

ALL_DAYS = TRAIN_DAYS + VAL_DAYS

CFG = dict(
    seq_len=24,
    batch_size=256,
    hidden_size=64,
    num_layers=2,
    cnn_channels=32,
    kernel_size=3,
    dropout=0.4,
    lr=1e-3,
    weight_decay=1e-4,
    n_epochs=80,
    patience=10,
    clip_norm=2.0,
)

DEVICE = (
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)