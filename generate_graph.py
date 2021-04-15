import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

# TODO Convert all as argument

parser = argparse.ArgumentParser(description='Graph Generator')
parser.add_argument('path', metavar='path', type=str, help='path location of JSON results')
parser.add_argument('--dest', metavar='dest', type=str, help='path location for image')

args = parser.parse_args()

dest = 'plot.png'
if args.dest: # default value
    dest = args.dest

with open(args.path, 'r') as f:
    lines = f.readlines()
    # Loss vs. iterations (epoch)
    loss = []
    acc = []
    for line in lines[1:]:
        d = json.loads(line)
        if d["mode"] == "train":
            loss.append(d["loss"])
            acc.append(d["top1_acc"])

    # TODO Add legend
    x = np.arange(len(loss))
    fig, ax1 = plt.subplots()
    fig.suptitle('Loss and Training Accuracy vs. iterations')
    ax1.set_xlabel('Number of iterations')
    ax1.set_ylabel('Loss')
    ax1.plot(x, loss, color='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Training accuracy')
    ax2.plot(x, acc, color='tab:red')
    plt.savefig(dest)