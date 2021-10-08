from torch import nn


class ModelWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.transformer)})"

    def forward(self, inputs):
        return self.transformer(**inputs)
