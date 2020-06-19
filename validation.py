#!/usr/bin/env python

import csv, os

target = "publicdataset.csv"

def validate(file):
    targetdict = {}
    with open(target, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        targetdict = dict((rows[0],rows[1]) for rows in reader)
    predicteddict = {}
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        predicteddict = dict((rows[0],rows[1]) for rows in reader)
    correct = 0
    total = len(targetdict)-1
    for i in range(total):
        if targetdict[str(i)] == predicteddict[str(i)]:
            correct += 1
        print("For {} target:{} Predicted:{}".format(i, targetdict[str(i)] , targetdict[str(i)]))
    print("Result {} Correct:{}/{}".format(correct/total, correct , total))