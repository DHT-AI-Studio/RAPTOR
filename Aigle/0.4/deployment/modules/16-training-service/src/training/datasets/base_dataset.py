from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable, Callable, List
from datasets import DatasetDict
import logging
from torch.utils.data import DataLoader
from datasets import DatasetDict, load_dataset
from pathlib import Path

from ..models.trainer_config import DatasetConfig


logger = logging.getLogger(__name__)


@runtime_checkable
class DataLoaderProtocol(Protocol):
    def __iter__(self): ...
    def __len__(self) -> int: ...


class BaseDataset(ABC):
    """
    Production-grade abstract base class for all dataset loaders.
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Any,
        **kwargs
    ):
        self.config = config
        self.tokenizer = tokenizer

        self.dataset: Optional[DatasetDict] = None
        self.raw_dataset: Optional[DatasetDict] = None
        self._tokenized = False

    @abstractmethod
    def _get_tokenize_fn(self) -> Callable:
        """
        Subclasses must implement this to return the specific tokenization/labeling function.
        """
        pass
    
    @abstractmethod
    def _get_default_collate_fn(self) -> Callable:
        """
        Subclasses must implement this to return their task-specific default DataCollator.
        """
        pass

    def _load_raw_data(self) -> DatasetDict:
        data_path = self.config.dataset_name_or_path
        target_train_split = self.config.train_split_name
        target_val_split = self.config.val_split_name

        if Path(data_path).exists():
            if data_path.endswith((".json", ".jsonl")):
                raw_data = load_dataset("json", data_files=data_path, split="train", cache_dir=self.config.cache_dir)
            else:
                raw_data = load_dataset("text", data_files=data_path, split="train", cache_dir=self.config.cache_dir)

            return DatasetDict({"train": raw_data})
        
        else:
            logger.info(f"Loading raw text dataset {data_path} from huggingface hub")
            dataset_dict = load_dataset(data_path, cache_dir=self.config.cache_dir)

            if not isinstance(dataset_dict, DatasetDict):
                raise TypeError(
                    f"Expected DatasetDict from {data_path}, but got {type(dataset_dict)}. "
                    f"Ensure you don't use the 'split' argument when loading the main dataset dict."
                )
            
            final_dataset_dict = DatasetDict()

            if target_train_split in dataset_dict:
                final_dataset_dict["train"] = dataset_dict[target_train_split]
            elif "train" in dataset_dict:
                 logger.info(f"Split '{target_train_split}' not found, using 'train' instead.")
                 final_dataset_dict["train"] = dataset_dict["train"]
            else:
                available_splits = list(dataset_dict.keys())
                if len(available_splits) == 1:
                    sole_split_name = available_splits[0]
                    logger.warning(
                        f"Dataset {data_path} does not have a 'train' split. "
                        f"Renaming sole split '{sole_split_name}' to 'train' for processing."
                    )
                    final_dataset_dict["train"] = dataset_dict[sole_split_name]
                else:
                    raise ValueError(
                        f"HuggingFace dataset {data_path} must contain a 'train' split. "
                        f"Available splits are: {available_splits}. Cannot proceed."
                        f"Use the 'train_split_name' argument to specify a different split name."
                    )
            
            if target_val_split:
                if target_val_split in dataset_dict:
                    final_dataset_dict["validation"] = dataset_dict[target_val_split]
                else:
                    logger.warning(
                        f"Specified validation split '{target_val_split}' not found in dataset. "
                        f"Available: {list(dataset_dict.keys())}. Validation set will be empty or split from train."
                    )
            else:
                logger.debug("Validation split name not specified. No existing validation split will be loaded.")            
            
            return final_dataset_dict

    def _split_and_limit_dataset(self, raw_dataset: DatasetDict) -> DatasetDict:
        """
        Split the validation set and limit the training set size.

        - Validation set split: only split 'train' into 'train' and 'validation' sets (Priority: val_ratio > val_size > no split).
        - Training set size limit: if train_size > 0, then limit the 'train' split size.
        """
        train = raw_dataset["train"]

        has_existing_val = "validation" in raw_dataset and len(raw_dataset["validation"]) > 0

        val_is_requested_by_ratio = self.config.val_ratio is not None and self.config.val_ratio > 0.0
        val_size_as_limit_is_requested = self.config.val_size is not None and self.config.val_size > 0
        
        # --- 1. Handling Validation Split ---
        if has_existing_val:
            # A. Existing Validation Split
            val = raw_dataset["validation"]
            logger.info(f"Using existing validation set ({len(val)} samples) from raw data.")

            # A.1. Check if val_ratio is set (conflict, warning, and ignore)
            if val_is_requested_by_ratio:
                logger.warning(
                    f"Validation splitting (val_ratio={self.config.val_ratio}) was ignored "
                    "because an existing validation split was found and prioritized. "
                    "The ratio parameter only triggers splitting from the training set."
                )
            
            # A.2. Check if val_size is set (no conflict, use for limiting)
            if val_size_as_limit_is_requested:
                val_limit_size = min(self.config.val_size, len(val))
                if val_limit_size < len(val):
                    # Limit the existing validation set size
                    val = val.shuffle(seed=self.config.seed).select(range(val_limit_size))
                    logger.info(f"Existing validation set size limited to {val_limit_size} samples by val_size.")
                else:
                    logger.info(f"Existing validation set size is {len(val)}, no need to limit by val_size.")

        elif val_is_requested_by_ratio or val_size_as_limit_is_requested:
            # B. No existing split, and either ratio or size is requested

            # B.1. Prioritize val_ratio for splitting
            if val_is_requested_by_ratio:
                if val_size_as_limit_is_requested:
                    logger.warning(
                        "Both val_ratio and val_size are set. val_ratio will take precedence for splitting the validation set."
                    )

                test_split_param = self.config.val_ratio
                logger.info(f"Splitting train dataset using val_ratio: {self.config.val_ratio}")
            
            # B.2. Use val_size for splitting if no ratio is set
            elif val_size_as_limit_is_requested:
                test_split_param = self.config.val_size
                logger.info(f"Splitting train dataset using absolute count val_size: {self.config.val_size}")
            
            # Perform splitting
            split = train.train_test_split(test_size=test_split_param, seed=self.config.seed)
            train = split["train"]
            val = split["test"]

        else:
            # C. No existing split, and no request for splitting or limiting
            val = train.select(range(0)) 
            logger.info("Validation split skipped as neither ratio nor size was requested.")
                
        # --- 2. Limit Training set size ---
        if self.config.train_size is not None and self.config.train_size > 0:
            train_size = min(self.config.train_size, len(train))
            train = train.shuffle(seed=self.config.seed).select(range(train_size))
            logger.info(f"Train set limited to {train_size} samples.")
        else:
            logger.info("No limit applied to train set size.")

        return DatasetDict({"train": train, "validation": val})

    def _filter_empty_samples(self, dataset_dict: DatasetDict) -> DatasetDict:
        """
        Filters out samples that cannot contribute meaningfully to the loss function.
        A sample is considered invalid if its 'labels' column does not contain 
        the EOS token ID, implying no complete model response was generated or marked.
        """
        filtered_dict = DatasetDict()

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            logger.warning("Tokenizer has no EOS token ID. Falling back to simple '-100' check.")
            filter_condition = lambda x: any(label != -100 for label in x['labels'])
        else:
            filter_condition = lambda x: eos_id in [label for label in x['labels'] if label != -100]

        for split_name, dataset in dataset_dict.items():
            if len(dataset) == 0:
                filtered_dict[split_name] = dataset
                continue

            # Perform filtering
            initial_count = len(dataset)
            
            # Filter out samples where all labels are -100.
            filtered_dataset = dataset.filter(
                filter_condition,
                desc=f"Filtering {split_name} (checking for valid loss token in labels)"
            )
            
            removed_count = initial_count - len(filtered_dataset)
            
            if removed_count > 0:
                logger.warning(
                    f"Removed {removed_count} samples from {split_name} split. "
                    f"Reason: Labels did not contain a valid EOS token (ID: {eos_id}) or were all -100. "
                    f"Remaining: {len(filtered_dataset)}"
                )
            
            filtered_dict[split_name] = filtered_dataset

        return filtered_dict

    def _tokenize_dataset(self, processed_dataset: DatasetDict) -> DatasetDict:
        """
        Subclasses get tokenization function and apply it to train/val splits.
        """
        tokenize_fn = self._get_tokenize_fn() # Get tokenization function from subclass
        
        train = processed_dataset["train"]
        val = processed_dataset["validation"]

        # Determine whether to use batched map based on configuration
        is_batched = self.config.batched_map 

        # Tokenize Train Set
        train_tokenized = train.map(
            tokenize_fn,
            batched=is_batched,
            remove_columns=train.column_names,
            desc="Tokenizing train",
        )
        
        # Tokenize Validation Set (only operate on non-empty sets)
        if len(val) > 0:
            val_tokenized = val.map(
                tokenize_fn,
                batched=is_batched,
                remove_columns=val.column_names,
                desc="Tokenizing validation",
            )
        else:
            val_tokenized = val
        
        tokenized_dict = DatasetDict({"train": train_tokenized, "validation": val_tokenized})
        
        return self._filter_empty_samples(tokenized_dict)

    def load_and_split(self) -> DatasetDict:
        """
        Public method: Load → Split → Tokenize.
        Returns tokenized DatasetDict with 'train' and 'validation' splits.
        """
        logger.info("Loading and tokenizing dataset...")
        self.raw_dataset = self._load_raw_data()
        processed = self._split_and_limit_dataset(self.raw_dataset)
        self.dataset = self._tokenize_dataset(processed)
        self._tokenized = True

        logger.info(
            f"Dataset ready: {len(self.dataset['train'])} train, "
            f"{len(self.dataset['validation'])} validation"
        )
        return self.dataset

    def get_dataloader(
        self,
        split: str = "train",
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False
    ) -> Optional[DataLoaderProtocol]:
        """ 
        Return a DataLoader for the specified split. 
        General logic for all dataset types.
        """
        self._ensure_tokenized()
        
        if split not in self.dataset:
            logger.warning(f"Split '{split}' not found in dataset. Returning None.")
            return None
            
        if len(self.dataset[split]) == 0:
            logger.warning(f"Split '{split}' found but is empty. Returning None.")
            return None
        
        dataset = self.dataset[split]

        if collate_fn is None:
            collate_fn = self._get_default_collate_fn()
        
        if 'labels' in dataset.column_names:
            dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        else:
            dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        if num_workers == 0  or num_workers is None:
            persistent_workers = False
        else:
            persistent_workers = True
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=drop_last
        )

    def _ensure_tokenized(self) -> None:
        if not self._tokenized:
            raise RuntimeError(
                "Dataset not tokenized. Call load_and_split() first."
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_length={self.config.max_length}, "
            f"cache_dir={self.config.cache_dir}, "
            f"tokenized={self._tokenized})"
        )

    def display_tokenized_sample(
        self, 
        input_ids: List[int], 
        labels: List[int],
        attention_mask: List[int], 
        token_limit: Optional[int] = None
    ) -> None:
        """
        Display the decoded conversation and label masking results for a tokenized sample.
        Useful for debugging any generation model that produces input_ids and labels.
        """
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            logger.error("Tokenizer not available to display sample.")
            return

        logger.info("--- Tokenized Sample Debug ---")
        
        # 1. Decode text
        try:
            # Use skip_special_tokens=False to preserve all special tokens for debugging chat template structure
            decoded_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            logger.info("Decoded Conversation (Full Sequence):")
            # Use repr() to display possible line breaks and special characters
            logger.info(repr(decoded_text)) 
        except Exception as e:
            logger.warning(f"Failed to decode token IDs to text for debug: {e}")

        # 2. Token-level label masking check
        logger.info("--- Token-Level Label Masking Check ---")
        
        tokens_full = self.tokenizer.convert_ids_to_tokens(input_ids)
        debug_output = []
        
        for idx, (input_id, label, mask, token) in enumerate(zip(input_ids, labels, attention_mask, tokens_full)):
            status = '🟢 Train' if label != -100 else '⚪ Loss Mask '
            attention = '🟢 Attention' if mask == 1 else '⚪ Not Attention'
            token_str = repr(token)
            label_str = f"{label:6d}" if label != -100 else "-100"
            debug_output.append(f"{idx:4d} | {status} | ID:{input_id:6d} | Label:{label_str} | {attention} | Token: {token_str}")

        logger.info(f"Total Sequence Length: {len(input_ids)}")

        if token_limit is None or token_limit >= len(debug_output):
            preview = debug_output
            display_limit = "All"
        else:
            preview = debug_output[:token_limit]
            display_limit = f"Top {token_limit}"

        logger.info(f"Label Masking Preview ({display_limit} tokens):\n" + "\n".join(preview))

        if token_limit is not None and len(debug_output) > token_limit:
            logger.info("... [Remaining tokens truncated for brevity]")