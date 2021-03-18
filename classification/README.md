## Classification
In this task, we want to classify whiskey reviews as real or fake using the dataset we generated in [the generation task](../generation).

There are three approaches as outlined in [the final reeport](../report/final/final.pdf)
* Training GPT2 as a classifier starting from a **public checkpoint**. This is implemented in the [gpt2_public](gpt2_public) directory alongside its metrics.
* Training GPT2 as a classifier starting from the **generation checkpoint**. Implementation in the [gpt2_from_checkpoint](gpt2_from_checkpoint) directory with metrics.
* Tracking the **Preplexity** of reviews on the language model. Implemented in the [perplexity](perplexity) directory.

