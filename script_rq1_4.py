import os
import _thread
import time

threshold = [0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]

for threshold in threshold:
    dir = 'train/resnet56/'
    des_dir = dir + '{}_modules.pkl'.format(threshold)
    commend = 'python modularity.py --base_dir {} --threshold {}'.format(dir,threshold)

    if os.path.exists(des_dir):
        print("exit file")
        continue;
    else:
        commend = 'python modularity.py --base_dir {} --threshold {}'.format(dir,threshold)
        os.system(commend)
        time.sleep(2)

