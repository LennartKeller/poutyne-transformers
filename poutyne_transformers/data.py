
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from transformers import default_data_collator


class TransformerCollator:
    def __init__(
        self, y_keys: Union[str, List[str]] = None, custom_collator: Callable = None
    ):
        self.y_keys = y_keys
        self.custom_collator = (
            custom_collator if custom_collator is not None else default_data_collator
        )

    def __call__(self, inputs: Tuple[Dict]) -> Tuple[Dict, Any]:
        batch_size = len(inputs)
        batch = self.custom_collator(inputs)
        if self.y_keys is None:
            y = torch.tensor(float("nan")).repeat(batch_size)
        elif isinstance(self.y_keys, list):
            # If we want to compute the loss later on we can remove the labels from input since we do not need the original loss.
            y = {
                key: batch.pop(key) if "labels" in key else batch.get(key)
                for key in self.y_keys
            }
        else:
            y = batch.get(self.y_keys)

        return batch, y
