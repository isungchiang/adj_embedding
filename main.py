import os
import math
import random
import zipfile
import numpy as np
import tensorflow as tf
import spacy
import collections
import gensim
import submission as submission

input_dir = './BBC_Data.zip'
data_file = submission.process_data(input_dir)
# data_file = 'data_file.json'


## Output file name to store the final trained embeddings.
embedding_file_name = 'adjective_embeddings.txt'

## Fixed parameters
num_steps = 100001
embedding_dim = 200


## Train Embeddings, and write embeddings in "adjective_embeddings.txt"
submission.adjective_embeddings(data_file, embedding_file_name, num_steps, embedding_dim)



model_file = 'adjective_embeddings.txt'
devset = 'dev_set.zip'
ground_truth={}
with zipfile.ZipFile(devset, "r") as zipf:
    for file in zipf.namelist():
        if file[-1]=='/' or '._' in file :continue
        with zipf.open(file, 'r') as f:
            split_index = file.index('/')
            name = file[split_index+1:]
            ground_truth[name] = [tf.compat.as_str(line.strip()) for line in f]
total_hit = 0
count = 0
for k,v in ground_truth.items():
    top_k = 100
    hit = 0
    output = []
    output = submission.Compute_topk(model_file, k, top_k)
    print(output)
    for i in output:
        if i in v:
            hit += 1
    count +=1
    total_hit += hit
    print('Hits@k({}) = {}'.format(k,hit))
    print('average hit= {}'.format(total_hit/count))
    print('total hit= {}'.format(total_hit))
