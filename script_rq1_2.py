import os


g = 0

dir = 'train/resnet56/'
des_dir = dir + 'cka_within_model_256.pkl'
print(des_dir)
if os.path.exists(des_dir):
    exit(112)
else:
    commend = 'CUDA_VISIBLE_DEVICES={} python analysis.py --experiment_dir {} --gpu {}'.format(g,dir,0)
    print(commend)
    os.system(commend)