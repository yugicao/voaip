#!/usr/bin/env python3

import json
import os
import torch
import torch.nn.functional as F
import librosa
from importlib import import_module

def load_config(cfg_path: str) -> dict:
    """Read JSON config and return dict."""
    with open(cfg_path, 'r') as f:
        return json.load(f)

def get_model(model_cfg: dict, device: torch.device) -> torch.nn.Module:
    """Model initial architecture in config"""
    module = import_module(f"models.{model_cfg['architecture']}")
    ModelClass = getattr(module, "Model")
    model = ModelClass(model_cfg).to(device)
    return model

def load_weights(model: torch.nn.Module, weights_path: str, device: torch.device) -> None:
    """Load state_dict from file .pth to model."""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Cannot find weights at {weights_path}")
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

def infer_one(audio_path: str, model: torch.nn.Module, device: torch.device) -> float:
    """
    - Đọc audio với librosa để hỗ trợ nhiều định dạng.
    - Chuyển stereo → mono, resample về 16 kHz.
    - Chạy qua model, lấy logits, softmax để ra xác suất lớp spoof (index 1).
    """
    wav, sr = librosa.load(audio_path, sr=None)
    
    # wav = librosa.effects.preemphasis(wav, coef=1.0)
    # wav = librosa.util.normalize(wav)
    # wav, _ = librosa.effects.trim(wav, top_db=30)

    wav = librosa.effects.preemphasis(wav, coef=1.0)
    wav = librosa.util.normalize(wav)

    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sr}Hz")

    x = torch.from_numpy(wav).float().unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        _, logits = model(x)
        probs = F.softmax(logits, dim=1)
    # trả về score của lớp 'spoof' (hoặc genuine tùy định nghĩa config)
    return probs[:, 1].item()

class AntiSpoofing:
    """
    Lớp gói gọn toàn bộ pipeline:
    - init: load config, build model, load weights
    - predict: trả về score và label
    """
    def __init__(self,
                 config_path: str = "./config/AASIST.conf",
                 weights_path: str = None,
                 device: str = None):
        # 1) Load config
        cfg = load_config(config_path)
        self.threshold = cfg.get('threshold', 0.5)
        model_cfg = cfg['model_config']
        # 2) Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        # 3) Build & load model
        self.model = get_model(model_cfg, self.device)
        ckpt = weights_path or cfg.get('model_path')
        load_weights(self.model, ckpt, self.device)

    def predict(self, audio_path: str) -> dict:
        """
        Chạy inference 1 file, trả về:
        {
          'score': float,    # xác suất lớp 1
          'label': str       # 'genuine' hoặc 'spoof'
        }
        """
        score = infer_one(audio_path, self.model, self.device)
        label = "genuine" if score >= self.threshold else "spoof"
        return {'score': score, 'label': label}
