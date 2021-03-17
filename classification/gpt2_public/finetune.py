"""
Copyright (C) 2021 Akram Sbaih, Stanford University
    You can contact the author at <akram at stanford dot edu>
This file is to finetune GPT2 for the whiskey reviews classification into real and fake using the generated dataset.
We want to prompt the model with the review to generate a token indicating whether it's fake or real.
"""
import os
import pandas as pd
from tqdm import tqdm

# load the reals and the fakes
reals = [pd.read_csv(f, usecols=['text']) for f in ["../../dataset/finetune/val.csv", "../../dataset/finetune/test.csv"]]
reals = pd.concat(reals, ignore_index=True)
reals = reals.apply(lambda row: row['text'][row['text'].find("<review>")+len("<review>"):], axis=1, result_type='broadcast')
reals['gt'] = 'real'
fakes = [pd.read_csv(f) for f in ["../../generation/generations_val.csv", "../../generation/generations_test.csv"]]
fakes = pd.concat(fakes, ignore_index=True).drop(columns=['prompt']).rename({'gen': 'text'})
fakes['gt'] = 'fake'
dataset = pd.concat([reals, fakes], ignore_index=True).sample(frac=1, replace=False, random_state=314)
samples_count = dataset.shape[0]

# initialize the formatted dataset (formatted for GPT2)
formatted_dataset = pd.DataFrame(columns=["text"])

# convert the dataset row by row
for _, row in tqdm(dataset.iterrows(), desc="Formatting reviews", total=samples_count):
    formatted = f"<review>{row['text']}<pred>{row['gt']}<|endoftext|>"
    formatted_dataset = formatted_dataset.append({'text': formatted}, ignore_index=True)

# split the dataset to train/valid/test
ftrain, fval, ftest = 0.90, 0.06, 0.04
assert ftrain + fval + ftest == 1
ctrain, cval = int(ftrain * samples_count), int(fval * samples_count)
ctest = samples_count - ctrain - cval
if not os.path.exists("../../dataset/classify_public"): os.mkdir("../../dataset/classify_public")
formatted_dataset[:ctrain].to_csv("../../dataset/classify_public/train.csv")
formatted_dataset[ctrain:ctrain+cval].to_csv("../../dataset/classify_public/val.csv")
formatted_dataset[-ctest:].to_csv("../../dataset/classify_public/test.csv")
print(f"The data is split into Train: {ctrain}, Val: {cval}, Test: {ctest} " 
      f"\n And saved in the ../../dataset/classify_public directory.")

# Now, finetune GPT2 on this dataset
# Set --nproc_per_node to the number of gpus
train_path = os.path.abspath("../../dataset/classify_public/train.csv")
val_path, test_path = train_path.replace("train.csv", "val.csv"), train_path.replace("train.csv", "test.csv")
finetuned_model_path = os.path.abspath("finetuned-model")
if not os.path.exists(finetuned_model_path): os.mkdir(finetuned_model_path)
command = f"cd ../../transformers/examples/language-modeling ; " \
          f"python -m torch.distributed.launch --nproc_per_node 2 run_clm.py --output_dir " \
          f"../../../classification/gpt2_public/finetuned-model-train/ --model_type gpt2 --model_name_or_path gpt2 " \
          f"--validation_file {val_path} --do_eval --do_train --train_file {train_path} " \
          f"--per_device_eval_batch_size 2 --per_device_train_batch_size 2 --num_train_epochs 5 --evaluation_strategy " \
          f"epoch --logging_steps 4 --logging_dir ../../../classification/gpt2_public/finetued-model-logs " \
          f"--save_strategy epoch --dataloader_num_workers 4 "
print(command)
# os.system(command)
print("Run the command above to start training.")

