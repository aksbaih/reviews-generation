"""
Copyright (C) 2021 Akram Sbaih, Stanford University
    You can contact the author at <akram at stanford dot edu>
This script helps you calculate precision, recall, and accuracy of the results of the classifier.
"""

import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_file", type=str, default="preds_test_latest.csv")
parser.add_argument("--gt_file", type=str, default="../../dataset/classify_public/test.csv")
args = parser.parse_args()

# Find ground truth
gt_raw = pd.read_csv(args.gt_file)
gt = np.array([1 if raw.endswith("<pred>fake<|endoftext|>") else 0 for raw in gt_raw['text']])  # True being fake

# Find predictions
pred_raw = pd.read_csv(args.pred_file)
pred = np.array([1 if raw == 'fake' else 0 for raw in pred_raw['gen']])  # True being fake

# run metrics
accuracy = np.sum(pred == gt) / gt.shape[0]
percision = np.sum(pred * gt) / np.sum(pred == 1)
recall = np.sum(pred * gt) / np.sum(gt == 1)

print("N, Accuracy, Percision, Recall: ", gt.shape[0], accuracy, percision, recall)
