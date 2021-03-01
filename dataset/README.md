## Whiskey Reviews Dataset

Use [scraping_script.py](scraping_script.py) to pull review data from [whiskyadvocate.com](https://www.whiskyadvocate.com). The script will store the dataset in the file [`data.csv`](data.csv) as a Pandas Dataframe with the following columns:

* whiskey: the name of the whiskey
* rating: int out of 100
* price: int in dollars
* review: text review

The dataset is not divided into training/validation/testing divisions as this design choice is left for the application.
