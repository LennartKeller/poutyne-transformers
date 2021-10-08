from typing import Any, Dict


def model_loss(outputs: Dict[str, Any], targets: Any) -> float:
    """
    Returns the loss of the huggingface transformers model.
    """
    return outputs["loss"]
