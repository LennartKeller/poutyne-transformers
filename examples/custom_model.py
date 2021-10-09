import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from datasets import Value
from torch.utils.data import DataLoader
from torch import optim
from poutyne import Model, Lambda
from poutyne_transformers import TransformerCollator, ModelWrapper

print("Loading model & tokenizer.")
transformer = AutoModel.from_pretrained(
    "distilbert-base-cased", output_hidden_states=True
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")


custom_model = nn.Sequential(
    ModelWrapper(transformer),
    Lambda(lambda outputs: outputs["last_hidden_state"][:, 0, :]),  # get the cls token
    nn.Linear(in_features=transformer.config.hidden_size, out_features=1),
    Lambda(lambda out: out.reshape(-1)),
)

print("Loading & preparing dataset.")
dataset = load_dataset("imdb")
dataset = dataset.map(
    lambda entry: tokenizer(
        entry["text"], add_special_tokens=True, padding="max_length", truncation=True
    ),
    batched=True,
)
dataset = dataset.remove_columns(["text"])
for split in ("train", "test"):
    new_features = dataset[split].features.copy()
    new_features["label"] = Value("float64")
    dataset[split] = dataset[split].cast(new_features)
dataset = dataset.shuffle()
dataset.set_format("torch")


collate_fn = TransformerCollator(y_keys="labels", remove_labels=True)
train_dataloader = DataLoader(dataset["train"], batch_size=16, collate_fn=collate_fn)
test_dataloader = DataLoader(dataset["test"], batch_size=16, collate_fn=collate_fn)


print("Preparing training.")
optimizer = optim.AdamW(custom_model.parameters(), lr=5e-5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(
    custom_model, optimizer, loss_function=nn.BCEWithLogitsLoss(), device=device
)

print("Starting training.")
model.fit_generator(train_dataloader, test_dataloader, epochs=1)
