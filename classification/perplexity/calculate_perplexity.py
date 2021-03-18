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

for seq in tqdm(reals.iterrows()):
    encodings = tokenizer(seq['text'], return_tensors='pt')
    print(encodings)
    seq = encodings.to(args.device)
    labels = seq.clone()
    output = model(seq, labels=labels)
    print(output.detach().cpu())
    perplexity = torch.exp(output[0])
    print(perplexity.detach().cpu())

