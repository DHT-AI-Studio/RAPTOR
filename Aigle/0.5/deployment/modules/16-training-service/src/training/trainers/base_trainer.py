"""
Abstract base trainer class defining the interface for all trainer implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ..models.trainer_config import TrainerConfig


class BaseTrainer(ABC):
    """
    Abstract base class for all trainer implementations.
    """

    def __init__(self, config: TrainerConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.training_state = {
            "epoch": 0,
            "global_step": 0,
            "best_metric": float('inf'),
            "is_training": False,
            "resume_checkpoint_path": None,
        }

    @abstractmethod
    def train(
        self,
        model: Any,
        train_dataset: Any, # Expects DataLoader or DataModule
        eval_dataset: Any | None = None, # Expects DataLoader or DataModule
        callbacks: list[Callable] | None = None
    ) -> dict[str, Any]:
        """
        Execute training loop.
        """

    @abstractmethod
    def validate(self, model: Any, eval_dataset: Any) -> dict[str, float]:
        """
        Run validation loop.
        """

    @abstractmethod
    def save_checkpoint(self, output_dir: str, epoch: int | None = None, step: int | None = None) -> str:
        """
        Save training checkpoint.
        """

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Set the checkpoint path for training resumption.
        The actual loading happens during the next call to train().

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.training_state["resume_checkpoint_path"] = checkpoint_path

    @abstractmethod
    def get_training_state(self) -> dict[str, Any]:
        """
        Get current training state.
        """

    def stop_training(self) -> None:
        """Stop training gracefully."""
        self.training_state["is_training"] = False