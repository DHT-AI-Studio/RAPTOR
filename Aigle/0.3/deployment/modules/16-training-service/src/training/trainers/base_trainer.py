"""
Abstract base trainer class defining the interface for all trainer implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable

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
        eval_dataset: Optional[Any] = None, # Expects DataLoader or DataModule
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Execute training loop.
        """
        pass

    @abstractmethod
    def validate(self, model: Any, eval_dataset: Any) -> Dict[str, float]:
        """
        Run validation loop.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, output_dir: str, epoch: int = None, step: int = None) -> str:
        """
        Save training checkpoint.
        """
        pass

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Set the checkpoint path for training resumption.
        The actual loading happens during the next call to train().

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.training_state["resume_checkpoint_path"] = checkpoint_path

    @abstractmethod
    def get_training_state(self) -> Dict[str, Any]:
        """
        Get current training state.
        """
        pass

    def stop_training(self) -> None:
        """Stop training gracefully."""
        self.training_state["is_training"] = False