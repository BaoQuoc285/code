import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn import svm
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch 
from datasets import load_dataset
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from preprocessing import preprocessing

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = load_dataset('csv', data_files='train1.csv', delimiter=',')

model_new = AutoModelForSequenceClassification.from_pretrained("vinai/phobert-base", num_labels = 3)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
small_train_dataset = tokenized_datasets["train"].select(range(4500))
small_eval_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(500))
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=2)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=1)

optimizer = AdamW(model_new.parameters(), lr=1e-4)
num_epochs = 15
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_new.to(device)

progress_bar = tqdm(range(num_training_steps))
num_epochs = 15
model_new.train()
for epoch in range(num_epochs):
    total_loss = 0
    i = 0
    for batch in train_dataloader:
        # print(batch.items())
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model_new(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        i+=1
        total_loss += loss
    print("EP Loss:", total_loss/i)

f = open("/content/data_phase_2.txt")
data_test  = f.readlines()
data_test_arr = []
for line in data_test:
  data_test_arr.append(preprocessing(line))

label_map = {
    0: 'negative',
    1: 'neutral',
    2: 'positive'
}

import numpy as np
model_new.eval()
model_new.to("cpu")
res = []
for data in data_test_arr:
  input_ids = torch.tensor([tokenizer.encode(data)])

  with torch.no_grad():
      out = model_new(input_ids).logits.softmax(dim=-1)
      print(out)
      lb = np.argmax(out)
      # lb = np.argmax(out.logits.softmax(dim=-1))
      print(lb.tolist())
      res.append(label_map[lb.tolist()])

with open("output3.txt", "w") as txt_file:
  for line in res:
    txt_file.write("".join(line) + "\n")