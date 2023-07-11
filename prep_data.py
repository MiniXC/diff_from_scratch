import json

from datasets import load_dataset
from speech_collator import SpeechCollator
from vocex import Vocex
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np
import torch

dataset = load_dataset("cdminix/libritts-r-aligned")

phone2idx = json.load(open("data/phone2idx.json"))
speaker2idx = json.load(open("data/speaker2idx.json"))

vocex_model = Vocex.from_pretrained("cdminix/vocex").model.to("cpu")

collator = SpeechCollator(
    speaker2idx=speaker2idx,
    phone2idx=phone2idx,
    use_speaker_prompt=True,
    overwrite_max_length=True,
    vocex_model=vocex_model,
    overwrite_cache=True,
)

dl_train = DataLoader(
    dataset["train"],
    batch_size=8,
    collate_fn=collator.collate_fn,
    num_workers=96,
    shuffle=True,
)

dl_val = DataLoader(
    dataset["dev"],
    batch_size=8,
    collate_fn=collator.collate_fn,
    num_workers=96,
    shuffle=True,
)

min_vals = []
max_vals = []
all_vals = []
mean_vals = []
std_vals = []
i = 0

# for item in tqdm(dl_train):
#     item["mel"] = torch.clamp(item["mel"], min=-10, max=0)
#     item["mel"] = (item["mel"] + 5) / 5
#     min_vals.append(item["mel"].min())
#     max_vals.append(item["mel"].max())
#     all_vals.append(item["mel"].flatten())
#     mean_vals.append(item["mel"].mean())
#     std_vals.append(item["mel"].std())
#     i += 1
#     if i == 1_000:
#         break

for item in tqdm(dl_train):
    item["vocex"] = torch.clamp(item["vocex"], min=-3, max=5)
    # BRING to -1 to 1 range
    item["vocex"] = ((item["vocex"] + 3) / 8) * 2 - 1
    min_vals.append(item["vocex"].min())
    max_vals.append(item["vocex"].max())
    all_vals.append(item["vocex"][item["phone_mask"]].flatten())
    mean_vals.append(item["vocex"].mean())
    std_vals.append(item["vocex"].std())
    i += 1
    if i == 1_000:
        break

# get percentiles of all values
all_vals = np.concatenate(all_vals)
percentiles = np.array([0.001, 0.01, 0.05, 0.1, 0.9, 0.95, 0.99, 0.999])
percentile_vals = np.quantile(all_vals, percentiles)
print(percentile_vals)

# all values histogram
plt.hist(all_vals, bins=100)
plt.savefig("all_vals.png")
plt.clf()
plt.hist(min_vals, bins=100)
plt.savefig("min_vals.png")
plt.clf()
plt.hist(max_vals, bins=100)
plt.savefig("max_vals.png")
plt.clf()
plt.hist(mean_vals, bins=100)
plt.savefig("mean_vals.png")
plt.clf()
plt.hist(std_vals, bins=100)
plt.savefig("std_vals.png")
plt.clf()

# for item in tqdm(dl_val):
#     pass