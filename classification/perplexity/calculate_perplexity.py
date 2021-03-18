"""
Copyright (C) 2021 Akram Sbaih, Stanford University
    You can contact the author at <akram at stanford dot edu>
This script helps you calculate the perplexity of strings giveen a language model
It is inspired by the huggingface https://huggingface.co/transformers/perplexity.html
"""
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import pandas as pd
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='gpt2')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()


tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
config = AutoConfig.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        ).to(args.device)

# load the reals and the fakes
reals = [pd.read_csv(f, usecols=['text']) for f in ["../../dataset/finetune/val.csv", "../../dataset/finetune/test.csv"]]
reals = pd.concat(reals, ignore_index=True)
reals = reals.apply(lambda row: row['text'][row['text'].find("<review>")+len("<review>"):], axis=1, result_type='broadcast')

fakes = [pd.read_csv(f, usecols=['gen']) for f in ["../../generation/generations_val.csv", "../../generation/generations_test.csv"]]
fakes = pd.concat(fakes, ignore_index=True).rename(columns={'gen': 'text'})

real_ps, fake_ps = [], []

for _, seq in tqdm(reals.iterrows(), total=reals.shape[0], desc="Evaluating Reals"):
    encodings = tokenizer(seq['text'], return_tensors='pt')
    seq = encodings.input_ids.to(args.device)
    labels = seq.clone()
    output = model(seq, labels=labels)[0]
    perplexity = torch.exp(output)
    real_ps.append(perplexity.item())

for _, seq in tqdm(fakes.iterrows(), total=fakes.shape[0], desc="Evaluating Fakes"):
    encodings = tokenizer(seq['text'], return_tensors='pt')
    seq = encodings.input_ids.to(args.device)
    labels = seq.clone()
    output = model(seq, labels=labels)[0]
    perplexity = torch.exp(output)
    fake_ps.append(perplexity.item())

with open('reals.txt', 'w') as f:
    f.writelines([str(itm) for itm in real_ps])

with open('fakes.txt', 'w') as f:
    f.writelines([str(itm) for itm in fake_ps])

