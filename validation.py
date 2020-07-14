#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, os

target = "publicdataset.csv"

def validate(predicteddict):
    targetdict = {}
    with open(target, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        targetdict = dict((rows[0],rows[1]) for rows in reader)
    correct = 0
    total = len(targetdict)-1
    for i in range(total):
        if targetdict[str(i)] == predicteddict[i]:
            correct += 1
        #print("For {} target:{} Predicted:{}".format(i, targetdict[str(i)] , targetdict[str(i)]))
    print("")
    print("Result {} Correct:{}/{}".format(correct/total, correct , total))
    return correct/total

def validatePath(path):
    predicteddict = {}
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        predicteddict = dict((rows[0],rows[1]) for rows in reader)
    validate(predicteddict)
    


#validate("output/predictions/prediction_100_0.0001_CNN3D_2020_06_29_18_00_46.csv")