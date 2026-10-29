import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import lightgbm as lgb
import numpy as np
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

class SimpleScaler:
    def __init__(self, mean, scale):
        self.mean_ = np.array(mean, dtype=np.float32)
        self.scale_ = np.array(scale, dtype=np.float32)
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

@dataclass
class ScalpSignal:
    action: str          # BUY | SELL | HOLD
    strategy: str        # LGBM | SCALPNET | DETERMINISTIC
    confidence: float    # 0.0-1.0
    sl_distance: float   # FIXO
    tp_distance: float   # FIXO
    predicted_mfe: float = 0.0 # Log only
    predicted_mae: float = 0.0 # Log only

SL_MULT = 1.0
TP_MULT = 1.2

class ScalpNet(nn.Module):
    def __init__(self, n_features=34, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(0.1),
        )
        self.cls_head = nn.Linear(hidden // 2, 3)
        self.reg_head = nn.Linear(hidden // 2, 2)

    def forward(self, x):
        h = self.shared(x)
        # Enforce non-negative excursion predictions (MAE >= 0, MFE >= 0) in ATR units
        return self.cls_head(h), F.softplus(self.reg_head(h))

class ScalpSignalEngine:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler: Optional[SimpleScaler] = None
        self.lgbm_model = None
        self.scalpnet = None
        self._load_models()
        
    def _load_models(self):
        # Allow running from different directories
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "models"
        
        scaler_path = models_dir / f"{self.symbol}_scaler.json"
        lgbm_path = models_dir / f"{self.symbol}_lgbm.txt"
        nn_path = models_dir / f"{self.symbol}_scalpnet.pt"
        
        if scaler_path.exists():
            try:
                with open(scaler_path, "r") as f:
                    data = json.load(f)
                self.scaler = SimpleScaler(data["mean"], data["scale"])
            except Exception as e:
                logger.error(f"Failed to load scaler for {self.symbol}: {e}")
                
        if lgbm_path.exists() and self.scaler:
            try:
                self.lgbm_model = lgb.Booster(model_file=str(lgbm_path))
                logger.info(f"[{self.symbol}] LightGBM loaded")
            except Exception as e:
                logger.error(f"Failed to load LGBM for {self.symbol}: {e}")
                
        if nn_path.exists() and self.scaler:
            try:
                n_feat = len(self.scaler.mean_) if self.scaler else 34
                self.scalpnet = ScalpNet(n_features=n_feat).to(self.device)
                self.scalpnet.load_state_dict(
                    torch.load(nn_path, map_location=self.device, weights_only=True)
                )
                self.scalpnet.eval()
                logger.info(f"[{self.symbol}] ScalpNet loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load ScalpNet for {self.symbol}: {e}")

    def predict(self, features: np.ndarray, atr: float) -> ScalpSignal:
        if self.scalpnet and self.scaler:
            return self._predict_scalpnet(features, atr)
        elif self.lgbm_model and self.scaler:
            return self._predict_lgbm(features, atr)
        else:
            return self._deterministic_fallback(features, atr)
            
    def _predict_lgbm(self, features: np.ndarray, atr: float) -> ScalpSignal:
        scaled = self.scaler.transform(features.reshape(1, -1))
        probs = self.lgbm_model.predict(scaled)[0]
        
        action_idx = int(np.argmax(probs))
        confidence = float(probs[action_idx])
        actions = ["BUY", "SELL", "HOLD"]
        
        return ScalpSignal(
            action=actions[action_idx],
            strategy="LGBM",
            confidence=confidence,
            sl_distance=atr * SL_MULT,
            tp_distance=atr * TP_MULT,
        )
        
    @torch.inference_mode()
    def _predict_scalpnet(self, features: np.ndarray, atr: float) -> ScalpSignal:
        scaled = self.scaler.transform(features.reshape(1, -1))
        x = torch.tensor(scaled, dtype=torch.float32).to(self.device)
        
        logits, sl_tp_raw = self.scalpnet(x)
        probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()
        
        action_idx = int(np.argmax(probs))
        confidence = float(probs[action_idx])
        actions = ["BUY", "SELL", "HOLD"]
        
        # Log only
        pred_mae = float(sl_tp_raw[0, 0].cpu()) * 3.0
        pred_mfe = float(sl_tp_raw[0, 1].cpu()) * 4.0
        
        return ScalpSignal(
            action=actions[action_idx],
            strategy="SCALPNET",
            confidence=confidence,
            sl_distance=atr * SL_MULT,
            tp_distance=atr * TP_MULT,
            predicted_mfe=pred_mfe,
            predicted_mae=pred_mae,
        )
        
    def _deterministic_fallback(self, features: np.ndarray, atr: float) -> ScalpSignal:
        # Fallback to simple logic if models are not loaded
        # Index 27 is trend_dir, 28 is regime_strength
        trend_dir = features[27]
        strength = features[28]
        
        action = "HOLD"
        confidence = 0.0
        
        if strength > 0.4:
            if trend_dir > 0.5:
                action = "BUY"
                confidence = 0.51 + strength * 0.1
            elif trend_dir < -0.5:
                action = "SELL"
                confidence = 0.51 + strength * 0.1
                
        return ScalpSignal(
            action=action,
            strategy="DETERMINISTIC",
            confidence=confidence,
            sl_distance=atr * SL_MULT,
            tp_distance=atr * TP_MULT,
        )
