from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Any

import torch
from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

@dataclass
class DataCollatorForMultiLabelClassification:
    """
    Data collator that will dynamically pad the inputs received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["input_ids"] = torch.tensor(batch["input_ids"], dtype=torch.int64)
        batch["attention_mask"] = torch.tensor(batch["attention_mask"], dtype=torch.int64)
        batch["coarse_labels"] = torch.tensor(batch["coarse_labels"], requires_grad=False, dtype=torch.float64)
        batch["fine_labels"] = torch.tensor(batch["fine_labels"], requires_grad=False, dtype=torch.float64)
        return batch
