import torch
from typing import Any, Dict


def model_loss(outputs: Dict[str, Any], targets: Any) -> torch.tensor:
    """Returns the internal loss of a transformers model.
    Make sure that labels are passed into the transformers.

    Args:
        outputs (Dict[str, Any]): Model ouput of the transformer model
        targets (Any): Pytounes labels, not used.

    Returns:
        torch.tensor: loss
    """

    return outputs["loss"]
