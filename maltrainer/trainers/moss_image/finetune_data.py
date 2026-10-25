"""The MOSS-Image finetune's data: the shared finetune data (maltrainer/trainers/finetune/dataset.py: regularisation
images, per-folder repeats, the prior captions, --suggest-steps) on the MOSS-Image previews. The first of the
finetune's mixins, ahead of the LoRA trainer it extends."""

from __future__ import annotations

from maltrainer.trainers.finetune.dataset import FinetuneDataset
from maltrainer.trainers.moss_image.previews import PreviewHooks
from maltrainer.training.advice import StepAdvice
from maltrainer.training.image_lora import ImageLoRATrainer


class FinetuneData(FinetuneDataset, PreviewHooks, StepAdvice, ImageLoRATrainer):
    pass
