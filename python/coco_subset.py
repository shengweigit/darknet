import os
import random

PREFIX = '/home/sw/workspace/darknet/dataset/val2014'
NUM_SUBSET = 5000
DATASET_PATH = '/home/sw/workspace/darknet/dataset/val2014'

files = os.listdir(DATASET_PATH)
subset = random.sample(files, NUM_SUBSET)
with open('coco_val_5k.list', 'w+') as f:
    for name in subset:
        f.write("%s\n" % os.path.join(PREFIX, name))

