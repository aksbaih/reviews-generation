This is my implementation of the first approach outlined in [the milestone](../../report/milestone/milestone.pdf). It uses the publicly available huggingface GPT2 checkpoint and fine-tunes it to classify fake and real reviews.

## Training
[`finetune.py`](finetune.py) preprocesses the real and fake reviews of the validation and test sets stored in [the dataset directory](../../dataset/finetune) and the [generations directory](../../generation).
The output is a formatted in csv's with the `text` column as `<review>{review text}<pred>{real / fake}<|endoftext|>`.

The dataset is split into train/val/test sets according to the provided ratio in `finetune.py` and stored as a subdirectory in [the dataset directory](../../dataset) called `classify_public`.

[`finetune.py`](finetune.py) generates a command at the end which you should run to start training using the huggingface framework.

## Run Predictions
Run the following command to find the prediction of the trained model on some file
```
python run_prediction.py \
    --model_type gpt2 \
    --model_name_or_path finetuned-model-train/ \
    --prompts ../dataset/classify_public/test.csv \
    --quite \
    --output_file pred_test.csv \
        2> /dev/null
```

## Run Metrics
You can run a few metrics on the predictions file. In this case, we find the Precision, Recall, and Accuracy
```
python run_metrics.py
```
