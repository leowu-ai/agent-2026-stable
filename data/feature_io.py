from pathlib import Path

import h5py
import numpy as np
import torch


def load_feature_file(path):
    """Load a 2D patch-feature matrix and optional patch coordinates."""
    path = Path(path)
    if path.suffix == ".npy":
        obj = np.load(path, allow_pickle=True)
        if getattr(obj, "shape", None) == () and obj.dtype == object:
            obj = obj.item()
        if isinstance(obj, dict):
            for key in ("feature", "features", "feats", "embeddings", "x"):
                if key in obj:
                    arr = obj[key]
                    break
            else:
                arr = next(v for v in obj.values() if hasattr(v, "shape") and len(v.shape) == 2)
            coords = obj.get("coords", obj.get("coordinates", obj.get("index")))
        else:
            arr, coords = obj, None
    elif path.suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("feature", "features", "feats", "embeddings", "x"):
                if key in obj:
                    arr = obj[key]
                    break
            else:
                arr = next(v for v in obj.values() if hasattr(v, "shape") and len(v.shape) == 2)
            coords = obj.get("coords", obj.get("coordinates", obj.get("index")))
        else:
            arr, coords = obj, None
        if torch.is_tensor(arr):
            arr = arr.numpy()
    elif path.suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as handle:
            key = next((k for k in ("features", "feature", "feats", "embeddings", "x") if k in handle), None)
            if key is None:
                key = next(k for k in handle.keys() if len(handle[k].shape) == 2)
            arr = handle[key][:]
            coords = handle["coords"][:] if "coords" in handle else None
    else:
        raise ValueError(f"Unsupported feature file: {path}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr, coords
