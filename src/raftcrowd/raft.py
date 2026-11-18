"""Lazy Torchvision RAFT adapter: RGB, explicit weights, pad then crop, one flow."""

from __future__ import annotations

import hashlib
import time
import numpy as np


class RaftEstimator:
    def __init__(self, config):
        # Import only during actual inference; feature/metric tests need no GPU stack.
        import torch
        import torchvision
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        self.torch = torch
        self.config = config
        self.device = torch.device(config["device"])
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Select a GPU runtime or explicitly set device=cpu.")
        if config["mixed_precision"] and self.device.type != "cuda":
            raise ValueError("mixed_precision requires CUDA in this implementation")
        torch.manual_seed(config["seed"])
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        weights = Raft_Large_Weights[config["weights"]]
        self.transform = weights.transforms()
        self.model = raft_large(weights=weights, progress=True).to(self.device).eval()
        parameter_hash = hashlib.sha256()
        for name, tensor in sorted(self.model.state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            parameter_hash.update(name.encode())
            parameter_hash.update(str(array.dtype).encode())
            parameter_hash.update(str(array.shape).encode())
            parameter_hash.update(array.tobytes())
        self.metadata = {
            "backend": "torchvision.raft_large", "weights": config["weights"],
            "weights_url": weights.url, "torch": torch.__version__,
            "state_dict_sha256": parameter_hash.hexdigest(),
            "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
            "device": str(self.device),
            "hardware": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
            "precision": "autocast_fp16" if config["mixed_precision"] else "fp32",
            "updates": config["updates"], "batch_size": 1,
            "timing_scope": "tensor upload + normalization + padding + RAFT + crop + CPU transfer; excludes decoding, resizing and CFI",
            "warmup": "none; per-pair timings include cold start and are diagnostic, not a deployment benchmark",
        }

    def __call__(self, first_rgb, second_rgb):
        torch = self.torch
        from torch.nn import functional as F

        if first_rgb.shape != second_rgb.shape or first_rgb.ndim != 3 or first_rgb.shape[2] != 3:
            raise ValueError("RAFT requires two equally sized HxWx3 RGB arrays")
        if first_rgb.dtype != np.uint8 or second_rgb.dtype != np.uint8:
            raise ValueError("RGB images must be uint8")
        h, w = first_rgb.shape[:2]
        if min(h, w) < 128:
            raise ValueError("RAFT-large requires image dimensions >=128")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        a = torch.from_numpy(np.ascontiguousarray(first_rgb)).permute(2, 0, 1)[None].to(self.device)
        b = torch.from_numpy(np.ascontiguousarray(second_rgb)).permute(2, 0, 1)[None].to(self.device)
        a, b = self.transform(a, b)
        # Right/bottom padding preserves the original pixel origin and vector units.
        pad = (0, (-w) % 8, 0, (-h) % 8)
        a, b = F.pad(a, pad, mode="replicate"), F.pad(b, pad, mode="replicate")
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.float16,
            enabled=self.config["mixed_precision"],
        ):
            predictions = self.model(a, b, num_flow_updates=self.config["updates"])
            flow = predictions[-1][0, :, :h, :w].float().permute(1, 2, 0).cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        if not np.isfinite(flow).all():
            raise ValueError("RAFT produced nonfinite flow")
        return flow, elapsed
