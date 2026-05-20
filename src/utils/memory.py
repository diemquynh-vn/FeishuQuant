import psutil
import torch
import gc

def ram_usage():
    try:
        m = psutil.virtual_memory()
        print(f'RAM used: {m.used/1e9:.1f} GB / {m.total/1e9:.1f} GB ({m.percent}%)')
        return m
    except:
        print("RAM usage: N/A")
        return None

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    return device

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()