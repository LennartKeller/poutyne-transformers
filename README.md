# poutyne-transformers

Train [🤗-transformers](https://huggingface.co/transformers/) models with [Poutyne](https://poutyne.org).

## Installation

```bash
pip install poutyne-transformers
```

## Example

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import optim
from poutyne import Model, Accuracy
from poutyne_transformers import (
    TransformerCollator,
    model_loss,
    ModelWrapper,
    MetricWrapper,
)

print("Loading model & tokenizer.")
transformer = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-cased", num_labels=2, return_dict=True
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")

print("Loading & preparing dataset.")
dataset = load_dataset("imdb")
dataset = dataset.map(
    lambda entry: tokenizer(
        entry["text"], add_special_tokens=True, padding="max_length", truncation=True
    ),
    batched=True,
)
dataset = dataset.remove_columns(["text"])
dataset = dataset.shuffle()
dataset.set_format("torch")

collate_fn = TransformerCollator(y_keys="labels")
train_dataloader = DataLoader(dataset["train"], batch_size=16, collate_fn=collate_fn)
test_dataloader = DataLoader(dataset["test"], batch_size=16, collate_fn=collate_fn)

print("Preparing training.")
wrapped_transformer = ModelWrapper(transformer)
optimizer = optim.AdamW(wrapped_transformer.parameters(), lr=5e-5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
accuracy = MetricWrapper(Accuracy(), pred_key="logits")
model = Model(
    wrapped_transformer,
    optimizer,
    loss_function=model_loss,
    batch_metrics=[accuracy],
    device=device,
)

print("Starting training.")
model.fit_generator(train_dataloader, test_dataloader, epochs=1)
```

You can also create models with a custom architecture using `torch.nn.Sequential` class:

```python
from torch import nn
from transformers import AutoModel
from poutyne import Lambda
from poutyne_transformers import ModelWrapper


transformer = AutoModel.from_pretrained(
    "distilbert-base-cased", output_hidden_states=True
)

custom_model = nn.Sequential(
    ModelWrapper(transformer),
    # Use distilberts [CLS] token for classification.
    Lambda(lambda outputs: outputs["last_hidden_state"][:, 0, :]),
    nn.Linear(in_features=transformer.config.hidden_size, out_features=1),
    Lambda(lambda out: out.reshape(-1)),
)
```
