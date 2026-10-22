"""The control API's request and response models: what Swagger shows."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from maltrainer.control.hub import CommandRecord, CommandState, RunState


class Action(str, Enum):
    """The commands that need no body (the console key each one stands for in brackets)."""

    SAVE = "save"  # [s]
    SAVE_AND_STOP = "save-and-stop"  # [q]
    PAUSE = "pause"  # [p]
    RESUME = "resume"  # [p again]
    GALLERY = "gallery"  # [g]


class CommandOut(BaseModel):
    id: int
    command: str
    state: CommandState
    detail: str = Field("", description="What happened, once the command is done or failed")
    submitted_at: float = Field(description="Unix time")
    settled_at: float | None = Field(None, description="Unix time the command finished; null while pending")

    @classmethod
    def of(cls, record: CommandRecord) -> CommandOut:
        return cls(
            id=record.id,
            command=record.command.value,
            state=record.state,
            detail=record.detail,
            submitted_at=record.submitted_at,
            settled_at=record.settled_at,
        )


class Window(BaseModel):
    """The training monitor's window: what the trainer logs as its ``[monitor]`` line, since the last one."""

    window_steps: int
    last_loss: float | None
    window_loss: float | None
    window_loss_bands: dict[str, float]
    window_prior: float | None
    seconds_per_step: float | None = Field(description="Without preview rounds and pauses")


class CapabilitiesOut(BaseModel):
    gallery: bool
    preview_settings: bool


class RunStatus(BaseModel):
    state: RunState
    step: int | None = Field(None, description="Optimizer steps done; null before the first step boundary")
    total_steps: int | None = None
    percent: float | None = None
    epoch: int | None = Field(None, description="1-based; null for trainers that do not track one")
    learning_rate: float | None = None
    eta_seconds: float | None = Field(None, description="An estimate: the window's seconds per step x steps left")
    window: Window | None = None
    published_seconds_ago: float | None = Field(
        None, description="Age of these numbers: they refresh once per optimizer step (not during a preview or save)"
    )
    capabilities: CapabilitiesOut


class PromptOut(BaseModel):
    slot: int = Field(description="The prompt's stable number: its seed offset and sample file name")
    prompt: str
    enabled: bool


class PreviewSettings(BaseModel):
    every: int = Field(description="Render previews every this many steps; 0 = off")
    rows: dict[str, bool] = Field(description="Which weights to render (raw, EMA), when the trainer has both")
    prompts: list[PromptOut]


class PromptToggle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slot: int
    enabled: bool


class PreviewSettingsPatch(BaseModel):
    """Only the fields you send change. Applied at the next step boundary, for this run only."""

    model_config = ConfigDict(extra="forbid")
    every: int | None = Field(None, ge=0)
    rows: dict[str, bool] | None = None
    prompts: list[PromptToggle] | None = None

    @model_validator(mode="after")
    def _something_to_change(self) -> PreviewSettingsPatch:
        if self.every is None and self.rows is None and self.prompts is None:
            raise ValueError("send at least one of every, rows, prompts")
        return self
