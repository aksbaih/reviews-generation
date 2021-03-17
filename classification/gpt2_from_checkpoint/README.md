This is my implementation of the first approach outlined in [the milestone](../../report/milestone/milestone.pdf). It uses the publicly available huggingface GPT2 checkpoint and fine-tunes it to classify fake and real reviews.

## Training
[`finetune.py`](finetune.py) preprocesses the real and fake reviews of the validation and test sets stored in [the dataset directory](../../dataset/finetune) and the [generations directory](../../generation).
The output is a formatted in csv's with the `text` column as `<review>{review text}<pred>{real / fake}<|endoftext|>`.

The dataset is split into train/val/test sets according to the provided ratio in `finetune.py` and stored as a subdirectory in [the dataset directory](../../dataset) called `classify_public`.

[`finetune.py`](finetune.py) generates a command at the end which you should run to start training using the huggingface framework.

## Note
***The only difference between this and [../gpt2_public](../gpt2_public) is that this launcher starts from [the checkpoint used for generation](../../generation) as outlined in the [milestone](../../report/milestone/milestone.pdf).***
