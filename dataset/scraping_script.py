"""
Copyright (C) 2021 Akram Sbaih, Stanford University
    You can contact the author at <akram at stanford dot edu>
This script helps you scrape review data from www.whiskyadvocate.com
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

PLACEHOLDER = "<placeholder>"
URL_TEMPLATE = f"https://www.whiskyadvocate.com/ratings-reviews/?search=&submit=+&brand_id=0&rating={PLACEHOLDER}&price=0&category=0&styles_id=0&issue_id=0"
# these replacements make sure that all reviews on the website are being scraped
PLACEHOLDER_REPLACEMENTS = ["95-100", "90-94", "80-89", "70-79", "60-69"]

dataset = pd.DataFrame(columns=["whiskey", "rating", "price", "review"])
accepted, rejected = 0, 0

for replacement in tqdm(PLACEHOLDER_REPLACEMENTS):
    url = URL_TEMPLATE.replace(PLACEHOLDER, replacement)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    reviews = soup.find_all('div', class_="ratings-grid-holder")[0].children
    for review in reviews:
        if isinstance(review, str): continue
        try:
            price = int(float(review.find_all('span', {'itemprop': 'price'})[0].text.replace(',', '')))
            text = review.find_all('div', class_='review-text')[0].find_all('p')[0].text
            rating = int(float(review.find_all('span', {'itemprop': 'ratingValue'})[0].text))
            whiskey = review.find_all('h1', {'itemprop': 'name'})[0].text
            dataset = dataset.append({"whiskey": whiskey, "rating": rating, "price": price, "review": text},
                                     ignore_index=True)
            accepted += 1
        except:
            rejected += 1

print(f"Accepted: {accepted} and rejected: {rejected}")
dataset.to_csv("data.csv")

