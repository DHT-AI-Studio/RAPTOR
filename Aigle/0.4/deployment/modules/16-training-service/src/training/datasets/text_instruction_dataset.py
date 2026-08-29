from transformers import PreTrainedTokenizer, default_data_collator, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from typing import Callable, Dict, Any, List, Optional
import logging
import json
import numpy as np

from .base_dataset import BaseDataset
from ..models.trainer_config import DatasetConfig


logger = logging.getLogger(__name__)


class TextInstructionDataset(BaseDataset):
    """
    A dataset that requires fine-tuning (SFT) and specific labeling for instruction tasks.
    Suitable for tasks such as SQuAD, Instruction Tuning, and Chat Format.
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        config.batched_map = True
        super().__init__(config, tokenizer)
        self._logged_first_sample = False

    def _get_field_with_fallback(
        self, 
        ex: Dict[str, Any], 
        logic_key: str, 
        fallback_keys: List[str] = []
    ) -> Optional[str]:
        """
        Retrieve the value of a specified key from a sample dictionary. If the key is not found, try a series of fallback keys.
        """
        # 1. Use the mapped key
        mapped_key = self.config.column_mapping.get(logic_key, logic_key) # Default to itself if not found
        if ex.get(mapped_key):
            return ex.get(mapped_key)
        
        # 2. Try the fallback keys
        for key in fallback_keys:
            # Make sure not to re-check the key that has already been checked by mapped_key
            if key != mapped_key and ex.get(key):
                return ex.get(key)
                
        return None

    def _get_output_text(self, ex: Dict[str, Any]) -> str:
        if ex.get("answers"): # SQuAD style
            answer_text = ex.get("answers", {}).get("text", [])
            return answer_text[0] if len(answer_text) > 0 else "The question cannot be answered from the context."
        
        output_text = self._get_field_with_fallback(
            ex, 
            logic_key="output", 
            fallback_keys=["answer", "output", "response"]
        )
        
        return output_text

    def _combine_reasoning_and_output(self, reasoning: Optional[str], output_text: str) -> str:
        if reasoning:
            logger.info("Reasoning found, combining with output text.")
            return f"<think>\n{reasoning}\n</think>\n\n{output_text}"
        else:
            return output_text

    def _build_default_messages(self, system_prompt, context, reasoning, input_text, output_text):
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if context:
            user_content = f"Context: {context}\n\nQuestion/Instruction: {input_text}"
        else:
            user_content = input_text

        assistant_content = self._combine_reasoning_and_output(reasoning, output_text)

        messages.extend([
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ])

        return messages

    def _validate_messages(self, msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Validate and clean a list of multi-turn conversation messages,
        including mapping common non-standard roles to standard roles.
        """
        # Define role mapping
        ROLE_MAPPING: Dict[str, str] = {
            "human": "user",
            "gpt": "assistant",
            "model": "assistant",
            "bot": "assistant",
            "system_prompt": "system"
        }

        valid_roles: List[str] = ["system", "user", "assistant"]
        
        cleaned_msgs: List[Dict[str, str]] = []
        for i, msg in enumerate(msgs):
            raw_role = str(msg.get("role")).lower()
            content = msg.get("content")
            role = ROLE_MAPPING.get(raw_role, raw_role)
            
            # 1. Check basic structure
            if not isinstance(msg, dict) or not role or not content:
                logger.warning(f"Skipping malformed message at index {i}. Message: {msg}")
                continue
            
            # 2. Check role validity
            if role not in valid_roles:
                logger.warning(f"Skipping message at index {i} due to unsupported role: {role}. Expected roles: {valid_roles}")
                continue

            # 3. Check content validity
            if not isinstance(content, str) or not content.strip():
                logger.warning(f"Skipping empty message at index {i} from role {role}.")
                continue
                
            # Ensure content is a string to avoid tokenizer errors
            msg["role"] = role
            msg["content"] = str(content)
            cleaned_msgs.append(msg)
            
        return cleaned_msgs

    def _prepare_messages(self, ex: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Unified method to extract or construct the 'messages' list.
        Supports both direct 'messages' column (Multi-turn) and legacy columns (Single-turn).
        """
        # 1. First, check if the sample is already in the single-turn or multi-turn conversation format (List[Dict])
        messages = self._get_field_with_fallback(ex, logic_key="messages")
        if messages and isinstance(messages, list):
            msgs = self._validate_messages(messages)
            if not msgs:
                raise ValueError("Dataset sample contains 'messages' column but all messages are invalid or empty after validation.")
            return msgs        
        
        # 2. Extract context, input
        context = self._get_field_with_fallback(ex, logic_key="context")
        reasoning = self._get_field_with_fallback(ex, logic_key="reasoning")
        input_text = self._get_field_with_fallback(
            ex, 
            logic_key="input", 
            fallback_keys=["question", "instruction", "input"]
        )        
        
        # 3. Check input_text, output_text validity
        if not input_text:
            raise ValueError(f"Dataset sample missing 'messages' or valid input/instruction/question field. Keys: {list(ex.keys())}")
        
        output_text = self._get_output_text(ex)
        if not output_text:
             raise ValueError("Dataset sample missing valid output field.")

        # 4. Construct messages
        system_prompt = self.config.default_system_prompt or ex.get("system_prompt")
        
        if system_prompt is not None:
             system_prompt = str(system_prompt).strip()
             if not system_prompt:
                 system_prompt = None
        
        return self._build_default_messages(system_prompt, context, reasoning, input_text, output_text)

    def _process_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        batch_size = len(next(iter(batch.values())))
        messages_batch = []
        
        # 1. Preparation stage: unify all samples to messages format
        for i in range(batch_size):
            ex = {k: batch[k][i] for k in batch.keys()}
            try:
                msgs = self._prepare_messages(ex)
                messages_batch.append(msgs)
            except ValueError as e:
                logger.error(f"Skipping sample {i} due to error: {e}")

        if not messages_batch:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        # 2. Batch Tokenization
        formatted_messages = self.tokenizer.apply_chat_template(conversation=messages_batch, tokenize=False)
        formatted_messages = [msg + self.tokenizer.eos_token for msg in formatted_messages]
        tokenized_full = self.tokenizer(
            formatted_messages, 
            truncation=True, 
            max_length=self.config.max_length,
            padding="max_length", 
            return_tensors=None,
            add_special_tokens=False
        )
        
        final_labels = []

        # 3. Multi-turn Masking Algorithm (The Firefly Approach)
        # We need to find out the token range in input_ids for each Assistant response
        for i in range(len(messages_batch)):
            full_ids = tokenized_full['input_ids'][i]
            msgs = messages_batch[i]
            
            # Initialize labels: set all to -100 (default to not train)
            current_labels = [-100] * len(full_ids)
            
            # Count the length of the prefix in tokens.
            current_len = 0

            for msg_idx, msg in enumerate(msgs):
                # Get the prefix of the conversation
                sub_msgs = msgs[:msg_idx+1]
                
                # Tokenize prefix                 
                add_generation_prompt = True if msg["role"] == "user" else False
                
                sub_tokens_dict = self.tokenizer.apply_chat_template(
                    sub_msgs, 
                    truncation=True, 
                    max_length=self.config.max_length, 
                    padding=False, 
                    return_dict=True,
                    return_tensors=None,
                    add_generation_prompt=add_generation_prompt 
                )
                sub_tokens = sub_tokens_dict['input_ids']
                next_len = len(sub_tokens)
                
                # If the conversation has already exceeded max_length, 
                # the following messages will be truncated.
                if current_len >= len(full_ids):
                    break
                
                # Fix edge case (when Truncation occurs)
                start_idx = current_len
                end_idx = min(next_len, len(full_ids))

                # If this message is generated by the Assistant, 
                # the labels of this segment should be equal to the Input IDs (for training purposes)
                if msg["role"] == "assistant":
                    if start_idx < end_idx:
                        current_labels[start_idx:end_idx] = full_ids[start_idx:end_idx]
                
                # Update current_len
                current_len = next_len

            try:
                # Ensure EOS token is also predicted
                eos_idx = full_ids.index(self.tokenizer.eos_token_id)
                current_labels[eos_idx] = full_ids[eos_idx]  
            except ValueError:
                pass  # EOS token not found, likely due to truncation

            final_labels.append(current_labels)

            # Debug Log
            if i == 0 and not self._logged_first_sample:
                self._log_multi_turn_debug(msgs, full_ids, current_labels, tokenized_full['attention_mask'][i])
                self._logged_first_sample = True

        tokenized_full['labels'] = final_labels
        return tokenized_full

    def _log_multi_turn_debug(self, msgs, full_ids, labels, attention_mask):
        logger.info(f"--- Tokenizing First Sample (Multi-turn Mode) ---")
        logger.info(f"Turns count: {len(msgs)}")
        logger.info(f"Raw Messages (Preview): {json.dumps(msgs, ensure_ascii=False, indent=2)}")
        self.display_tokenized_sample(full_ids, labels, attention_mask, token_limit=None)

    def _get_tokenize_fn(self) -> Callable:
        return self._process_batch

    def _get_default_collate_fn(self) -> Callable:
        return default_data_collator