# reviews-generation
Final project for Stanford CS224N NLP class involving fake reviews generation and detection

* The [Generation](generation) directory has implementation and documentation for fake whiskey reviews generation.
* The [Classification](classification) directory has implementation and documentation for classifying fake and real reviews using three different approaches.

All the approaches and their details are outlined in [the final report](report/final/final.pdf) of the project; please take a look :)


## Submodules
This repo contains the following submodule repos:
* [hugginface transformers](https://github.com/huggingface/transformers) used [here](transformers).

To clone correctly, use the following command
```
git clone --recurse-submodules https://github.com/aksbaih/reviews-generation
```

## Requirements
Install the requirements by running
```
pip install -r requirements.txt
```
You also need to install the requirements for huggingface gpt2 as mentioned in [their docs](transformers/examples/README.md)
```
cd transformers
pip install .
cd examples/language-modeling
pip install -r requirements.txt
cd ../text-generation
pip install -r requirements.txt
```
This repo also assumes that your machine has two GPUs with 12GB memory each. I used 2 K-80s on Azure.
