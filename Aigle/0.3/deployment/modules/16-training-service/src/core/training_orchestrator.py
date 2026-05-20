import logging
from typing import Dict, Any, Optional
from transformers import PreTrainedTokenizer

from ..training.trainers.lightning_trainer import LightningTrainer
from ..training.models.trainer_config import TrainerConfig, DatasetConfig
from ..training.datasets.text_instruction_dataset import TextInstructionDataset 
from ..training.datasets.text_dataset import TextDataset 


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Core orchestrator for executing a single training task (Data Load -> Train -> Save Model)"""
        
    def __init__(
        self,
        task_type: str,  # "qa", "text" 
        config: TrainerConfig,
        dataset_config: DatasetConfig,
        cancel_token: Any = None,
        mlflow_uri: Optional[str] = None,
    ):
        self.task_type = task_type
        self.config = config
        self.dataset_config = dataset_config
        self.trainer: Optional[LightningTrainer] = None
        self.dataset = None 
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.cancel_token = cancel_token
        self.mlflow_uri = mlflow_uri

    def _init_components(self):
        """Unified initialization of Trainer and Dataset components"""
        
        logger.info(f"Initializing components for task: {self.task_type}")

        # 1. Initialize Trainer
        self.trainer = LightningTrainer(self.config, cancel_token=self.cancel_token, mlflow_uri=self.mlflow_uri)
        self.tokenizer = self.trainer.tokenizer
        logger.info("LightningTrainer and Module initialized.")

        # 2. Initialize Dataset
        dataset_cls = TextInstructionDataset if self.task_type == "instruction" else TextDataset
        self.dataset = dataset_cls(tokenizer=self.tokenizer, config=self.dataset_config)
        logger.info(f"Dataset handler created: {dataset_cls.__name__}")
        
    def run(self) -> Dict[str, Any]:
        """Run the entire training process"""
        logger.info(f"Starting {self.task_type.upper()} training job...")
        
        # 1. Initialize components
        self._init_components()

        # 2. Load and Prepare DataLoader
        self.dataset.load_and_split()
        train_loader = self.dataset.get_dataloader(
            "train", 
            batch_size=self.config.batch_size, 
            num_workers=self.dataset_config.num_workers,
            drop_last=True
        )
        if train_loader is None:
            raise RuntimeError(
                "Failed to create DataLoader for the 'train' split. Training cannot proceed."
            )

        val_loader = self.dataset.get_dataloader(
            "validation", 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=self.dataset_config.num_workers
        )

        # 3. Start Training
        logger.info(f"Starting training with max_epochs={self.config.max_epochs}...")
        metrics = {}
        try:
            metrics = self.trainer.train(train_loader, eval_dataset=val_loader)        
            logger.info(f"Training finished: {metrics}")
        except Exception as e:
            logger.error(f"Orchestrator caught exception during training: {e}")
            raise
     
        return metrics