"""
Prior preservation for a full finetune: Differential Output Preservation (DOP, Ostris's AI Toolkit,
as Fizgig applies it to finetunes).

A LoRA's DOP compares the model with and without the adapter. A finetune has no adapter to switch
off, so the original model's answers are recorded once, before the first step:

1. for each training image whose caption names a subject, the caption with each trigger replaced
   by its class word ("photo of feliciaday, a woman ..." -> "photo of woman, a woman ...")
2. ``samples_per_image`` noised versions of the image (a sigma from the trainer's own sampler and
   a noise seed), and the original model's prediction on each, with that class caption

During training, a step on such an image adds ``weight * MSE(model(noisy, sigma, class caption),
recorded prediction)`` for one of its recorded inputs. Without the trigger, the model is held to
what it did before, so the subjects stay behind their trigger words instead of bleeding into the
class.

**Images added later** (a resumed run with new people) are recorded from the original model as
well: ``weights_from_file`` puts the trained parameters' original values back from the base
checkpoint for the recording and restores the trained ones bit for bit afterwards. Frozen
parameters never changed, so the model is then exactly the original.

Only the sigma and noise seed of each input are kept; the noise is regenerated from the seed on
the CPU, so it is identical every time. Predictions are kept in bf16 on the CPU (128 x 32 x 32 at
512 px for MOSS-Image: 0.26 MB each).

Noising is flow matching, ``(1 - sigma) x0 + sigma noise``, the convention of every trainer here.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from maltrainer.finetune.classes import class_caption, resolve_classes

__all__ = ["class_caption", "resolve_classes"]  # re-exported: train_moss_image_finetune.py imports them from here

logger = logging.getLogger(__name__)

# (x_t [1, ...] on device, sigma [1] on device, class caption) -> prediction [1, ...]
PredictFn = Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor]


CONTAINER_PREFIXES = ("", "diffusion_model.", "model.diffusion_model.")


@contextlib.contextmanager
def weights_from_file(
    named: Sequence[tuple[str, torch.nn.Parameter]],
    path: str | Path,
    prefixes: Sequence[str] = CONTAINER_PREFIXES,
) -> Iterator[None]:
    """The parameters hold the values stored under their names in the safetensors file at
    ``path`` (e.g. the base checkpoint) for the duration, then their own values again, bit for
    bit. Their current values wait in CPU RAM meanwhile (the trained set's size)."""
    backup = [param.detach().to("cpu", copy=True) for _, param in named]
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle, torch.no_grad():
            keys = set(handle.keys())
            for name, param in named:
                key = next((prefix + name for prefix in prefixes if prefix + name in keys), None)
                if key is None:
                    raise KeyError(f"{Path(path).name} has no tensor for {name}")
                param.copy_(handle.get_tensor(key).to(param.device, param.dtype))
        yield
    finally:
        with torch.no_grad():
            for (_, param), saved in zip(named, backup, strict=False):
                param.copy_(saved.to(param.device))


def _noise(seed: int, shape: torch.Size) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32)


class PriorPreservation:
    def __init__(self, weight: float = 1.0, samples_per_image: int = 8, every: int = 1, seed: int = 0):
        self.weight = float(weight)
        self.samples_per_image = max(int(samples_per_image), 1)
        self.every = max(int(every), 1)
        self.seed = int(seed)
        self.captions: dict[str, str] = {}  # key -> class caption
        self.sigmas: dict[str, torch.Tensor] = {}  # key -> [n] float32
        self.seeds: dict[str, torch.Tensor] = {}  # key -> [n] int64
        self.targets: dict[str, torch.Tensor] = {}  # key -> [n, ...] bf16, CPU

    def __contains__(self, key: str) -> bool:
        return key in self.targets

    def __len__(self) -> int:
        return len(self.targets)

    @staticmethod
    def noisy(x0: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        s = sigma.view(-1, *([1] * (x0.dim() - 1)))
        return (1.0 - s) * x0 + s * noise

    @torch.no_grad()
    def build(
        self,
        items: Iterable[tuple[str, torch.Tensor, str]],
        sample_sigma: Callable[[], torch.Tensor],
        predict: PredictFn,
        device: torch.device,
    ) -> None:
        """Record the original model's predictions. ``items``: (key, x0 [C, ...] on the CPU,
        class caption); call this before the first training step."""
        items = list(items)
        rng = random.Random(self.seed)
        for key, cpu_x0, caption in tqdm(items, desc="Prior: the original model's predictions"):
            x0 = cpu_x0.to(device, torch.float32).unsqueeze(0)
            sigmas, seeds, preds = [], [], []
            for _ in range(self.samples_per_image):
                sigma = sample_sigma().reshape(1).to(device, torch.float32)
                seed = rng.getrandbits(62)
                noise = _noise(seed, x0.shape).to(device)
                preds.append(predict(self.noisy(x0, noise, sigma), sigma, caption)[0].to("cpu", torch.bfloat16))
                sigmas.append(float(sigma))
                seeds.append(seed)
            self.captions[key] = caption
            self.sigmas[key] = torch.tensor(sigmas, dtype=torch.float32)
            self.seeds[key] = torch.tensor(seeds, dtype=torch.int64)
            self.targets[key] = torch.stack(preds)
        size = sum(t.numel() * t.element_size() for t in self.targets.values()) / 1024**2
        logger.info(
            "Prior: %d images x %d inputs recorded (%.0f MB in CPU RAM)",
            len(self.targets),
            self.samples_per_image,
            size,
        )

    def due(self, step: int) -> bool:
        """Whether optimizer step ``step`` (0-based) adds the prior term."""
        return self.weight > 0 and step % self.every == 0

    def loss(self, key: str, x0: torch.Tensor, predict: PredictFn) -> torch.Tensor | None:
        """MSE of the model now against the original on one recorded input of ``key`` (x0 [1, ...]
        on the device), or None when ``key`` has none. Uses Python's ``random``, so a resumed run
        restores it with the other RNG state."""
        if key not in self.targets:
            return None
        i = random.randrange(self.targets[key].shape[0])
        device = x0.device
        sigma = self.sigmas[key][i : i + 1].to(device)
        noise = _noise(int(self.seeds[key][i]), x0.shape).to(device)
        pred = predict(self.noisy(x0.float(), noise, sigma), sigma, self.captions[key])
        return torch.nn.functional.mse_loss(pred.float(), self.targets[key][i : i + 1].to(device).float())

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        """One safetensors file; a resumed run must load it, since the model it came from is gone."""
        keys = sorted(self.targets)
        tensors = {}
        for i, key in enumerate(keys):
            tensors[f"{i}.target"] = self.targets[key].contiguous()
            tensors[f"{i}.sigma"] = self.sigmas[key]
            tensors[f"{i}.seed"] = self.seeds[key]
        metadata = {
            "maltrainer_prior": json.dumps(
                {
                    "keys": keys,
                    "captions": [self.captions[k] for k in keys],
                    "samples_per_image": self.samples_per_image,
                }
            )
        }
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        save_file(tensors, str(tmp), metadata=metadata)
        tmp.replace(path)

    def load(self, path: str | Path) -> None:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            info = json.loads(handle.metadata()["maltrainer_prior"])
            saved = int(info.get("samples_per_image", self.samples_per_image))
            if saved != self.samples_per_image:
                logger.warning(
                    "Prior: %s has %d inputs per image; new images get %d", path, saved, self.samples_per_image
                )
            for i, (key, caption) in enumerate(zip(info["keys"], info["captions"], strict=False)):
                self.captions[key] = caption
                self.targets[key] = handle.get_tensor(f"{i}.target")
                self.sigmas[key] = handle.get_tensor(f"{i}.sigma")
                self.seeds[key] = handle.get_tensor(f"{i}.seed")
        logger.info("Prior: %d images' recorded predictions loaded from %s", len(self.targets), path)

    def keys(self) -> list[str]:
        return list(self.targets)
