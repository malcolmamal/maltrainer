"""Prior preservation's start for a full finetune (maltrainer/finetune/prior.py): the original model's predictions
recorded before the first step, or on resume reloaded from the state, with those of images added since recorded from
the original weights. A trainer brings the two model-specific parts: ``_prior_sigma`` (a training sigma for an image)
and ``_original_weights`` (the original model in place for a while, e.g. ``prior.weights_from_file``)."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch

from maltrainer.finetune.prior import PriorPreservation
from maltrainer.trainers.finetune.dataset import FinetuneDataset

logger = logging.getLogger(__name__)


class FinetunePriorSetup(FinetuneDataset):
    prior_state_name: ClassVar[str] = "prior.safetensors"  # in the resume state folder
    resume_dir: Path | None
    transformer: Any

    if TYPE_CHECKING:  # defined further along the finetune's MRO (its checkpoint part)

        def _name(self) -> str: ...

    def _prior_sigma(self, x0: torch.Tensor) -> torch.Tensor:
        """A training sigma for the image x0 [1, C, H, W] on the device."""
        raise NotImplementedError

    def _original_weights(self) -> contextlib.AbstractContextManager[Any]:
        """The original model's weights in place for the duration, the trained ones bit for bit again after."""
        raise NotImplementedError

    def _original_file(self) -> Path:
        """The base checkpoint ``_original_weights`` reads (named in the log)."""
        return Path(self.config["model"]["transformer_path"])

    def _prior_latent(self, key: str) -> torch.Tensor:
        """A sample's cached latent as [C, H, W] on the CPU (a trainer that caches it batched drops the axis)."""
        return self.latent_cache[key]

    def _predict(self, noisy: torch.Tensor, sigma: torch.Tensor, caption: str) -> torch.Tensor:
        context = self.text_cache[caption].to(self.device, self.dtype).unsqueeze(0)
        return cast(torch.Tensor, self.transformer(noisy.to(self.dtype), sigma, context))

    def _record_prior(self, captions: dict[str, str]) -> None:
        if self.prior is None:
            return
        self.prior.build(
            ((self._rel(key), self._prior_latent(key), caption) for key, caption in captions.items()),
            self._prior_sigma,
            self._predict,
            self.device,
        )

    def _setup_prior(self) -> None:
        """Record (or, on resume, reload) the original model's predictions for the prior term."""
        if self.prior is None:
            return
        captions = self._prior_captions()
        if not captions:
            raise ValueError(
                f"prior.weight > 0, but no caption names a trigger of prior.classes {sorted(self.prior_classes)}"
            )
        if self.resume_dir is not None:
            self._reload_prior(self.prior, self.resume_dir / self.prior_state_name, captions)
        else:
            self._record_prior(captions)
        self.prior_path = Path(self.config["output"]["path"]) / f"{self._name()}_prior.safetensors"
        self.prior_path.parent.mkdir(parents=True, exist_ok=True)
        self.prior.save(self.prior_path)
        self.run_metadata.update(
            {
                "prior_weight": self.prior.weight,
                "prior_samples_per_image": self.prior.samples_per_image,
                "prior_every": self.prior.every,
                "prior_classes": self.prior_classes,
                "prior_images": len(self.prior),
            }
        )
        logger.info(
            "Prior preservation: weight %g on %d images, every %d step(s), classes %s",
            self.prior.weight,
            len(self.prior),
            self.prior.every,
            self.prior_classes,
        )

    def _reload_prior(self, prior: PriorPreservation, path: Path, captions: dict[str, str]) -> None:
        """The saved predictions; those of images added since (e.g. new people) recorded from the original weights."""
        if not path.exists():
            raise FileNotFoundError(
                f"prior.weight > 0, but {path} is missing: the resumed run had no prior, and the "
                "original model it needs is gone. Set prior.weight: 0 to resume without one."
            )
        prior.load(path)
        missing = {k: c for k, c in captions.items() if self._rel(k) not in prior}
        if not missing:
            return
        logger.info(
            "Prior: %d image(s) added since it was recorded; recording theirs from the original "
            "weights in %s (~%.0f min)",
            len(missing),
            self._original_file().name,
            len(missing) * prior.samples_per_image / 6 / 60,
        )
        was_training = self.transformer.training
        self.transformer.eval()
        with self._original_weights():
            self._record_prior(missing)
        self.transformer.train(was_training)
        torch.cuda.empty_cache()
