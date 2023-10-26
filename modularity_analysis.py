import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse


z1 = []
z2 = []
z3 = []
z4 = []
z5 = []
z6 = []

divs = [10,25,50,75,100]

threshold = [0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]
commend = []
for threshold in threshold[::-1]:
    z5.append(threshold)
    dir = '/Users/heoseungbeom/code/Deep-Arc/RQ1/train/resnet56/{}_modules.pkl'.format(threshold)

    cka = pickle.load(tf.io.gfile.GFile(dir, 'rb'))

    cka[1] = cka[1]+1
    cka[3] = cka[3]+1
    l1 = 0
    b1 = 0
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for k in cka[0]:
        if k[0]!=k[1]:
            l1 += (k[1]-k[0]+1)

    for k in cka[2]:
        if k[0]!=k[1]:
            b1 += (k[1]-k[0]+1)

    print(l1/cka[1],(cka[1]-len(cka[0]))/cka[1],b1/cka[3],(cka[3]-len(cka[2]))/cka[3])
    z6.append(b1/cka[3])


ans_pkl = {'z5':z5,'z6':z6}
df = pd.DataFrame(ans_pkl)
df.to_pickle('ans.pkl')