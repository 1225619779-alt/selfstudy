import numpy as np
def _clip_prob(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1.0 - 1e-6)

def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip_prob(p)
    return np.log(p / (1.0 - p))

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))

def _fuse_posteriors(p_verify: np.ndarray, p_ddd: np.ndarray, *, verify_weight: float = 0.65) -> np.ndarray:
    wv = float(np.clip(verify_weight, 0.0, 1.0))
    wd = 1.0 - wv
    fused = _sigmoid(wv * _logit(p_verify) + wd * _logit(p_ddd))
    return np.clip(fused, 0.0, 1.0)
