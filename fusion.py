# ASSUMPTION Both models use the same annotation file. Remember to use the same TEST ANNOTATION FILE for RGB and FLOW if we removed videos from the flow annotation files.
# Input: JSON files (can be more than 2) of models we want to fuse results. (1 array of 101 predictions for each video)
# For each JSON output, pass it to a softmax function.

import argparse
import numpy as np
import sys
import json
from mmaction.datasets import build_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Multi-stream Fusion')
parser.add_argument('n_streams', metavar='N', type=int, help='number of streams')
parser.add_argument('paths', metavar='paths', type=str, nargs='+', help='path location of JSON results')
parser.add_argument('--weights', metavar='weights', type=float, nargs='+', help='weights for each stream')

args = parser.parse_args()

if args.n_streams != len(args.paths):
    print("Input error: Number of streams doesn't match number of paths provided.")
    sys.exit()

if args.weights and args.n_streams != len(args.weights):
    print("Input error: Number of weights doesn't match number of streams.")
    sys.exit()
          
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Load JSON results
streams = []
for path in args.paths:
    with open(path) as f:
        r = json.load(f)
        streams.append(r)
streams = np.array(streams)

# Apply softmax to all predictions in each stream
for i in range(len(streams)):
    for j in range(len(streams[i])):
        streams[i][j] = softmax(streams[i][j])

# Fuse streams
weights = np.ones(streams.shape[0])
if args.weights:
    weights = args.weights
fusion = np.average(streams, axis=0, weights=weights)

# TODO Manage to integrate with existing code. This will involve passing a configuration file.
# dataset = build_dataset(cfg.data.test, dict(test_mode=True))
# eval_res = dataset.evaluate(fusion)
# for name, val in eval_res.items():
#     print(f'{name}: {val:.04f}')

# Assumption: Use UCF101 dataset only
# TODO pass an input the annotation file used by both models
# Get the labels
annotation_file = 'data/YMJA/ymja_test_ben_split.txt' #'data/ucf101/ucf101_val_split_1_rawframes.txt'
labels = []

with open(annotation_file) as f:
    for line in f:
        labels.append(int(line.split()[2]))

labels = np.array(labels)
preds = np.argmax(fusion, axis=1)
if len(labels) != len(preds):
    print('Input error: Labels and predictions do not have the same lengths')
    sys.exit()

# TODO Can add confusion matrix or other
# Calculate accuracies

print('Top 1 accuracy', np.sum(labels == preds)/len(labels))

# micro: Calculate metrics globally by counting the total true positives, false negatives and false positives.
# macro: Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
# Micro-averaging better when suspect class imbalance
print('Precision (weighted)', precision_score(labels, preds, average='weighted'))
print('Recall (weighted)', recall_score(labels, preds, average='weighted'))
print('F1 Score (weighted)', f1_score(labels, preds, average='weighted'))

print(classification_report(labels, preds))

# Confusion matrix
# cf_matrix = confusion_matrix(labels, preds)
categories = ['No Penalty', 'Tripping', 'Hooking', 'Slashing', 'Holding']
# sns_plot = sns.heatmap(cf_matrix, annot=True, xticklabels=categories, yticklabels=categories)
# sns_plot.figure.savefig("cm.png")

# Report style confusion matrix
matrixConfusion = confusion_matrix(labels, preds)
df_cm = pd.DataFrame(matrixConfusion, categories, categories)
plt.figure(figsize=(3,3))
# sn.set(font_scale=1.6) # for label size
g = sns.heatmap(df_cm, annot=True,  cmap='coolwarm') # font size
g.set_yticklabels(categories, rotation=90)
plt.yticks(rotation=0)
plt.ylabel("True Values")
plt.xlabel("Predicted Values")
plt.xticks(rotation=45, ha='right')
g.figure.tight_layout()
plt.savefig("best_cm.png", dpi=750)
plt.subplots_adjust(left=.8)
plt.show()