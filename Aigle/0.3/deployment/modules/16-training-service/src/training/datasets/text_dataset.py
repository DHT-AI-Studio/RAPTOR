from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling
from typing import Callable
import logging

from .base_dataset import BaseDataset
from ..models.trainer_config import DatasetConfig


logger = logging.getLogger(__name__)


class TextDataset(BaseDataset):
    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        config.batched_map = True
        super().__init__(config, tokenizer)

    def _get_tokenize_fn(self) -> Callable:
        eos_token = self.tokenizer.eos_token
        text_column_name = self.config.column_mapping.get("text", "text")

        def tokenize_fn(examples):
            if text_column_name not in examples:
                raise KeyError(f"Missing required text column '{text_column_name}' in examples.")
            
            texts = examples[text_column_name]
            processed_texts = []
            
            for i, text in enumerate(texts):
                if not isinstance(text, str):
                    logger.warning(f"Skipping non-string data found in text column: {type(text)}")
                    continue
                if not text.strip().endswith(eos_token):
                    if i == 0:
                        logger.info("First text in dataset: " + text + eos_token)
                    processed_texts.append(text + eos_token)
                else:
                    processed_texts.append(text)

            tokenized = self.tokenizer(
                processed_texts,
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
                return_attention_mask=True,
                return_special_tokens_mask=True
            )
            return tokenized
        return tokenize_fn

    def _get_default_collate_fn(self) -> Callable:
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM
        )