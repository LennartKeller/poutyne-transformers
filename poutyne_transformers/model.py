from typing import Any, Dict
from torch import nn
from transformers import PreTrainedModel


class ModelWrapper(nn.Module):
    def __init__(self, transformer: PreTrainedModel):
        super().__init__()
        self.transformer = transformer

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.transformer)})"

    def forward(self, inputs) -> Dict[str, Any]:
        return self.transformer(**inputs)

    def save_pretrained(self, *args, **kwargs) -> None:
        self.transformer.save_pretrained(*args, **kwargs)
