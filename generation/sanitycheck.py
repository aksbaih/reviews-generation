"""
Copyright (C) 2021 Akram Sbaih, Stanford University
    You can contact the author at <akram at stanford dot edu>
This file is to test the setup for fine-tuning GPT2 on a simple task and see if it picks up.
"""
import os
import pandas as pd

# First, I generate a big csv file for training and one for validation of dummy data
if not os.path.exists("tmp"): os.mkdir("tmp")
dummy_training = ["dummy-data"] * 10000
training_df = pd.DataFrame(dummy_training, columns=["text"])
training_df.to_csv("tmp/dummy.csv")
dummy_path = os.path.abspath('tmp/dummy.csv')

dummy_eval = ["dummy-"] * 1000
eval_df = pd.DataFrame(dummy_eval, columns=["text"])
eval_df.to_csv("tmp/dummy-eval.csv")
dummy_eval_path = os.path.abspath("tmp/dummy-eval.csv")

# Now, finetune GPT to this dummy data
# You can stop this once GPT2 overfits which should happen fast
command = f"cd ../transformers/examples/language-modeling ; \
                python run_clm.py \
                    --model_name_or_path gpt2 \
                    --train_file {dummy_path} \
                    --validation_file {dummy_eval_path} \
                    --do_train \
                    --do_eval \
                    --output_dir /tmp/test-clm"
print(command)
os.system(command)
