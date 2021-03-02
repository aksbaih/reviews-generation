"""
Copyright (C) 2021 Akram Sbaih, Stanford University
    You can contact the author at <akram at stanford dot edu>
This file is to finetune GPT2 for the whiskey reviews dataset.
We want to prompt the model with the price, rating, and name of whiskey to generate a text review.
"""
import os
import pandas as pd
from tqdm import tqdm

# load the labeled dataset and shuffle it deterministically
labeled_data = pd.read_csv("../dataset/data.csv")
labeled_data = labeled_data.sample(frac=1, replace=False, random_state=314)
samples_count = labeled_data.shape[0]

# initialize the formatted dataset (formatted for GPT2)
formatted_dataset = pd.DataFrame(columns=["text"])

# convert the dataset row by row
for _, row in tqdm(labeled_data.iterrows(), desc="Formatting reviews", total=samples_count):
    formatted = f"<price>{row['price']}<rating>{row['rating']}<whiskey>{row['whiskey']}<review>{row['review']}"
    formatted_dataset = formatted_dataset.append({'text': formatted}, ignore_index=True)

# split the dataset to train/valid/test
ftrain, fval, ftest = 0.90, 0.06, 0.04
assert ftrain + fval + ftest == 1
ctrain, cval = int(ftrain * samples_count), int(fval * samples_count)
ctest = samples_count - ctrain - cval
if not os.path.exists("../dataset/finetune"): os.mkdir("../dataset/finetune")
formatted_dataset[:ctrain].to_csv("../dataset/finetune/train.csv")
formatted_dataset[ctrain:ctrain+cval].to_csv("../dataset/finetune/val.csv")
formatted_dataset[-ctest:].to_csv("../dataset/finetune/test.csv")
print(f"The data is split into Train: {ctrain}, Val: {cval}, Test: {ctest} " 
      f"\n And saved in the ../dataset/finetune directory.")

# Now, finetune GPT2 on this dataset
# Set --nproc_per_node to the number of gpus
train_path = os.path.abspath("../dataset/finetune/train.csv")
val_path, test_path = train_path.replace("train.csv", "val.csv"), train_path.replace("train.csv", "test.csv")
finetuned_model_path = os.path.abspath("finetuned-model")
if not os.path.exists(finetuned_model_path): os.mkdir(finetuned_model_path)
command =   f"cd ../transformers/examples/language-modeling ; \\" \
            f"python -m torch.distributed.launch --nproc_per_node 2 run_clm.py \\" \ 
            f"--model_name_or_path gpt2 \\" \
            f"--train_file {train_path} \\" \
            f"--validation_file {val_path} \\" \
            f"--do_train \\" \
            f"--do_eval \\" \
            f"--output_dir {finetuned_model_path} "
print(command)
# os.system(command)
print("Run the command above to start training.")
