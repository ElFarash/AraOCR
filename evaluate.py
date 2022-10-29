#!/usr/bin/python

from __future__ import print_function

import sys, os, editdistance

from utilities import calculate_wer, manual_calculate_wer, insert
import fastwer
import numpy as np
import pandas as pd

# Edit Distance (Levenshtein distance)
# in computational linguistics and computer science, edit distance is a string metric, 
# a way of quantifying how dissimilar two strings (e.g., words) are to one another, that is measured by counting 
# the minimum number of operations required to transform one string into the other. 

if len(sys.argv) != 3:
    sys.exit('USAGE: evalute.py ocr_output ref_text')

model_name = sys.argv[1].split('/')[1]
print(model_name)

columns = ['model', 'image_name', 'ref_text', 'ocr_output', 'WER' , 'CER' , 'Accuracy', 'Time Consumed']
experiments = pd.DataFrame(columns = columns)



distances = []
accuracies = []
wer_all = []


for file_name in os.listdir(sys.argv[1]):
    one_row = []
    with open(os.path.join(sys.argv[1], file_name), encoding='utf8') as f:
        predicted = ''.join(f.read())

    with open(os.path.join(sys.argv[2], file_name), encoding='utf8') as f:
        truth = ''.join(f.read())

    
    # calculate edit distance
    distance = editdistance.eval(predicted, truth)
    distances.append(distance)
    print(f'{file_name} edit distance: {distance}')


    # calculate accuracy
    acc = max(0, 1 - distance / len(truth))
    accuracies.append(acc)
    print(f'{file_name} acc: {acc}')


    # WER
    wer = calculate_wer(predicted, truth)
    print(f'{file_name} wer: {wer}')

    # # wer2
    # wer2 = manual_calculate_wer(predicted, truth)
    # print(f'{file_name} wer2: {wer2}')


    # wer using fastwer
    wer3 = fastwer.score_sent(predicted, truth, char_level=False)
    print(f'{file_name} wer3: {wer3}')

    # CER using fastwer
    cer = fastwer.score_sent(predicted, truth, char_level=True)
    print(f'{file_name} cer: {cer}')

    print('-------------------------------------------------------------')


    one_row.append(model_name)
    one_row.append(file_name.split('.')[0])
    one_row.append(truth)
    one_row.append(predicted)
    one_row.append(wer3)
    one_row.append(cer)
    one_row.append(acc)
    one_row.append('15')

    insert(experiments,one_row)





experiments.to_excel("test2.xlsx")



# print(f'Total distance = {sum(distances)}')
# print('Average Accuracy = %.2f%%' % (sum(accuracies) / len(accuracies) * 100))


