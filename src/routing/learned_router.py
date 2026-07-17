"""Implement learned expert routing from hidden-state features."""

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EXPERT_TO_IDX: Dict[str, int] = {
    'text': 0,
    'image': 1,
    'uml': 2,
    'general': 3,
}
IDX_TO_EXPERT: Dict[int, str] = {v: k for k, v in EXPERT_TO_IDX.items()}
ALL_EXPERTS: List[str] = ['text', 'image', 'uml', 'general']


class FocalLoss:
    """Calculate focal loss for learned routing."""

    def __init__(self, alpha=None, gamma: float = 2.0):
        """Initialize the instance."""
        self.alpha = alpha  # torch.Tensor or None
        self.gamma = gamma

    def __call__(self, logits, targets):
        """Invoke the object."""
        import torch
        import torch.nn.functional as F

        alpha = self.alpha
        if alpha is not None and alpha.device != logits.device:
            alpha = alpha.to(logits.device)
            self.alpha = alpha

        ce_loss = F.cross_entropy(logits, targets, weight=alpha, reduction='none')

        with torch.no_grad():
            p_t = torch.exp(-ce_loss)

        focal_weight = (1.0 - p_t) ** self.gamma

        return (focal_weight * ce_loss).mean()


class RouterMLP:
    """Predict expert probabilities from hidden-state features."""

    def __init__(
        self,
        input_dim: int = 4096,
        hidden1: int = 512,
        hidden2: int = 256,
        num_classes: int = 4,
        dropout1: float = 0.15,
        dropout2: float = 0.10,
    ):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise RuntimeError("RouterMLP requires PyTorch")

        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.block1 = nn.Sequential(
                    nn.Linear(input_dim, hidden1),
                    nn.LayerNorm(hidden1),
                    nn.GELU(),
                    nn.Dropout(dropout1),
                )
                self.block2 = nn.Sequential(
                    nn.Linear(hidden1, hidden2),
                    nn.LayerNorm(hidden2),
                    nn.GELU(),
                    nn.Dropout(dropout2),
                )
                self.skip = nn.Linear(hidden1, hidden2, bias=False)
                self.head = nn.Linear(hidden2, num_classes)

            def forward(self, x):
                h1 = self.block1(x)
                h2 = self.block2(h1) + self.skip(h1)
                return self.head(h2)

        self.model = _MLP().to(self.device)
        self.calibration_offsets: np.ndarray = np.zeros(num_classes, dtype=np.float32)
        self.num_classes = num_classes
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"RouterMLP initialized | parameters: {total_params:,} | device: {self.device} | "
            f"architecture: {input_dim}→{hidden1}→(residual)→{hidden2}→{num_classes}"
        )


    def save(self, path) -> None:
        """Save the current state."""
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'state_dict': self.model.state_dict(),
                'input_dim': self.input_dim,
                'hidden1': self.hidden1,
                'hidden2': self.hidden2,
                'calibration_offsets': self.calibration_offsets,
            },
            path,
        )
        cal_nonzero = np.any(self.calibration_offsets != 0)
        logger.info(
            f"Router saved: {path} "
            f"(input_dim={self.input_dim}, hidden1={self.hidden1}, hidden2={self.hidden2}, "
            f"calibrated={'yes' if cal_nonzero else 'no'})"
        )

    def load(self, path) -> bool:
        """Load saved state."""
        import torch
        path = Path(path)
        if not path.exists():
            logger.error(f"Router weight file does not exist: {path}")
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)

            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
                ckpt_input_dim = ckpt.get('input_dim', self.input_dim)
                ckpt_hidden1 = ckpt.get('hidden1', 512)
                ckpt_hidden2 = ckpt.get('hidden2', 128)
            else:
                state_dict = ckpt
                ckpt_input_dim = self.input_dim
                ckpt_hidden1, ckpt_hidden2 = self.hidden1, self.hidden2
                logger.warning(
                    f"Router weights use the legacy bare state_dict format; "
                    f"assuming input_dim={self.input_dim}, hidden1={self.hidden1}, hidden2={self.hidden2}"
                )

            if (ckpt_input_dim != self.input_dim
                    or ckpt_hidden1 != self.hidden1
                    or ckpt_hidden2 != self.hidden2):
                logger.warning(
                    f"Architecture mismatch (current: {self.input_dim}/{self.hidden1}/{self.hidden2}, "
                    f"checkpoint: {ckpt_input_dim}/{ckpt_hidden1}/{ckpt_hidden2}); rebuilding MLP..."
                )
                self.__init__(
                    input_dim=ckpt_input_dim,
                    hidden1=ckpt_hidden1,
                    hidden2=ckpt_hidden2,
                )

            self.model.load_state_dict(state_dict)
            self.model.eval()

            if isinstance(ckpt, dict) and 'calibration_offsets' in ckpt:
                self.calibration_offsets = np.array(
                    ckpt['calibration_offsets'], dtype=np.float32
                )
            else:
                self.calibration_offsets = np.zeros(self.num_classes, dtype=np.float32)

            cal_nonzero = np.any(self.calibration_offsets != 0)
            logger.info(
                f"Router loaded: {path} "
                f"(input_dim={self.input_dim}, hidden1={self.hidden1}, hidden2={self.hidden2}, "
                f"calibrated={'yes, offsets=' + str(self.calibration_offsets.round(2)) if cal_nonzero else 'no'})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load router: {e}")
            return False


    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict expert probabilities."""
        import torch
        import torch.nn.functional as F

        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).to(self.device)
            logits = self.model(x)

            if np.any(self.calibration_offsets != 0):
                offsets_t = torch.tensor(
                    self.calibration_offsets, dtype=torch.float32, device=logits.device
                )
                logits = logits + offsets_t

            probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def set_calibration_offsets(self, offsets: np.ndarray) -> None:
        """Set probability-calibration offsets."""
        self.calibration_offsets = np.array(offsets, dtype=np.float32)
        logger.info(
            f"Calibration offsets set: "
            f"text={offsets[0]:+.2f}, image={offsets[1]:+.2f}, "
            f"uml={offsets[2]:+.2f}, general={offsets[3]:+.2f}"
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict an expert assignment."""
        probs = self.predict_proba(features)
        return np.argmax(probs, axis=1)

    def predict_names(self, features: np.ndarray) -> List[str]:
        """Predict expert names."""
        indices = self.predict(features)
        return [IDX_TO_EXPERT[int(i)] for i in indices]

    def predict_top2(
        self, features: np.ndarray, collapse_threshold: float = 0.85
    ) -> List[Tuple[str, str, float, float]]:
        """Predict the top two experts."""
        probs = self.predict_proba(features)
        results = []

        for prob in probs:
            top2_idxs = np.argsort(prob)[::-1][:2]
            e1, e2 = IDX_TO_EXPERT[top2_idxs[0]], IDX_TO_EXPERT[top2_idxs[1]]
            w1, w2 = float(prob[top2_idxs[0]]), float(prob[top2_idxs[1]])

            if w1 >= collapse_threshold:
                results.append((e1, e1, 1.0, 0.0))
            else:
                w_sum = w1 + w2
                results.append((e1, e2, w1 / w_sum, w2 / w_sum))

        return results

    def get_routing_stats(self, features: np.ndarray) -> Dict[str, int]:
        """Return routing stats."""
        names = self.predict_names(features)
        stats = {e: 0 for e in ALL_EXPERTS}
        for name in names:
            stats[name] += 1
        return stats


class HiddenStateExtractor:
    """Extract routing features from hidden states."""

    def __init__(self, base_model, tokenizer, max_length: int = 512):
        """Initialize the instance."""
        self.model = base_model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def extract(
        self,
        inputs: List[str],
        batch_size: int = 4,
        normalize: bool = True,
    ) -> np.ndarray:
        """Extract routing features."""
        import torch

        self.model.eval()
        all_features = []
        total = len(inputs)

        for i in range(0, total, batch_size):
            batch = inputs[i: i + batch_size]
            batch_features = self._extract_batch(batch)
            all_features.append(batch_features)

            if (i // batch_size) % 20 == 0:
                logger.info(f"  Feature extraction progress: {min(i + batch_size, total)}/{total}")

        features = np.concatenate(all_features, axis=0)

        if normalize:
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            features = features / norms

        logger.info(f"Feature extraction complete: shape={features.shape}, normalized={normalize}")
        return features

    def _extract_batch(self, batch: List[str]) -> np.ndarray:
        """Extract batch."""
        import torch

        try:
            encoded = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = encoded['input_ids'].to(self.model.device)
            attention_mask = encoded['attention_mask'].to(self.model.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = outputs.hidden_states[-1]

                mask = attention_mask.unsqueeze(-1).float()      # (B, seq_len, 1)
                sum_hidden = (last_hidden * mask).sum(dim=1)     # (B, hidden_size)
                sum_mask = mask.sum(dim=1).clamp(min=1e-9)       # (B, 1)
                batch_features = (sum_hidden / sum_mask).cpu().float().numpy()  # (B, hidden_size)

            return batch_features

        except Exception as e:
            logger.error(f"  Batch feature extraction failed: {e}")
            hidden_size = self.model.config.hidden_size
            return np.zeros((len(batch), hidden_size), dtype=np.float32)

    def extract_and_save(
        self,
        inputs: List[str],
        save_path,
        labels: Optional[List[int]] = None,
        batch_size: int = 4,
    ) -> np.ndarray:
        """Extract and save."""
        features = self.extract(inputs, batch_size=batch_size)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if labels is not None:
            np.savez(save_path, features=features, labels=np.array(labels, dtype=np.int64))
        else:
            np.savez(save_path, features=features)

        logger.info(f"Features saved: {save_path} (features={features.shape})")
        return features


class LearnedRouterInference:
    """Run learned-router inference from a saved checkpoint."""

    def __init__(
        self,
        base_model,
        tokenizer,
        router_ckpt,
        input_dim: int = 4096,
        max_length: int = 512,
        collapse_threshold: float = 0.85,
        feature_cache_path: Optional[str] = None,
    ):
        """Initialize the instance."""
        self.extractor = HiddenStateExtractor(base_model, tokenizer, max_length)
        self.router = RouterMLP(input_dim=input_dim)
        self.collapse_threshold = collapse_threshold
        self.feature_cache_path = Path(feature_cache_path) if feature_cache_path else None

        if not self.router.load(router_ckpt):
            raise RuntimeError(f"Failed to load router weights: {router_ckpt}")

        logger.info(
            f"LearnedRouterInference initialized | "
            f"input_dim={self.router.input_dim} | "
            f"collapse_threshold={collapse_threshold}"
        )

    def route_single(self, input_text: str) -> str:
        """Route one input."""
        features = self.extractor.extract([input_text], batch_size=1)
        return self.router.predict_names(features)[0]

    def route_batch(self, inputs: List[str], batch_size: int = 4) -> List[str]:
        """Route a batch of inputs."""
        features = self._get_features(inputs, batch_size)
        return self.router.predict_names(features)

    def get_ensemble_weights(
        self, inputs: List[str], batch_size: int = 4
    ) -> List[Tuple[str, str, float, float]]:
        """Return ensemble weights."""
        features = self._get_features(inputs, batch_size)
        return self.router.predict_top2(features, self.collapse_threshold)

    def get_routing_probs(
        self, inputs: List[str], batch_size: int = 4
    ) -> np.ndarray:
        """Return routing probs."""
        features = self._get_features(inputs, batch_size)
        return self.router.predict_proba(features)

    def _get_features(self, inputs: List[str], batch_size: int) -> np.ndarray:
        """Return features."""
        if self.feature_cache_path and self.feature_cache_path.exists():
            try:
                data = np.load(self.feature_cache_path)
                features = data['features']
                if len(features) == len(inputs):
                    logger.debug(f"Features loaded from cache: {self.feature_cache_path}")
                    return features
            except Exception:
                pass
        return self.extractor.extract(inputs, batch_size=batch_size)


def load_router_from_checkpoint(
    base_model,
    tokenizer,
    ckpt_path: str = "checkpoints/exp10_learned_router/router_mlp_best.pt",
    **kwargs,
) -> LearnedRouterInference:
    """Load router from checkpoint."""
    return LearnedRouterInference(
        base_model=base_model,
        tokenizer=tokenizer,
        router_ckpt=ckpt_path,
        **kwargs,
    )
