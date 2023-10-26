import os
import _thread
import time

dir = 'train/resnet56/'
des_dir = dir + 'cka_within_model_256_b.pkl'
if os.path.exists(des_dir):
    exit(112)
else:
    commend = 'CUDA_VISIBLE_DEVICES={} python cka_extract.py --base_dir {} --gpu {}'.format(0,dir,4096)
    os.system(commend)
    time.sleep(5)
  
