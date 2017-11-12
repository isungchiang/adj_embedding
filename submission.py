## Submission.py for COMP6714-Project2
###################################################################################################################
import os
import math
import random
import zipfile
import numpy as np
import tensorflow as tf
import spacy
import collections
import json
import gensim

# Tunable Parameters:
batch_size = 100
num_sample = 2000
learning_rate = 0.02
skip_window = 7
vocabulary_size = 8000


def train(sentence, word_dict, num_steps, embedding_dim,embeddings_file_name):
    sentence_num = 0
    train_num = 0
    _batch_inputs = []
    _batch_labels = []

    graph = tf.Graph()

    with graph.as_default():
        with tf.device('/cpu:0'):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dim], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            _weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_dim],
                                                       stddev=1.0 / math.sqrt(embedding_dim)))
            _biases = tf.Variable(tf.zeros([vocabulary_size]))
            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=_weights, biases=_biases,
                                                 labels=train_labels, inputs=embed,
                                                 num_sampled=num_sample, num_classes=vocabulary_size))
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            # Add variable initializer.
            init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:
        session.run(init)
        while (train_num < num_steps):
            current_sentence = sentence[sentence_num]
            # update sentence id
            sentence_num = (sentence_num + 1) % len(sentence)
            # generate batch
            for i in range(len(current_sentence)):
                start = max(0, i - skip_window)
                end = min(len(current_sentence) - 1, i + skip_window + 1)
                for context in range(start, end):
                    # in skip-gram model, we need to skip the target word
                    if context == i: continue
                    if (not current_sentence[i] in word_dict) or \
                            (not current_sentence[context] in word_dict): continue
                    _batch_inputs.append(word_dict[current_sentence[i]])
                    _batch_labels.append(word_dict[current_sentence[context]])
                    # reach the size of mini batch
                    if (len(_batch_inputs) == batch_size):
                        _batch_inputs = np.array(_batch_inputs, dtype=np.int32)
                        _batch_labels = np.array(_batch_labels, dtype=np.int32)
                        _batch_labels = np.reshape(_batch_labels, [batch_size, 1])

                        feed_dict = {train_inputs: _batch_inputs, train_labels: _batch_labels}
                        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)

                        train_num = train_num+ 1
                        if train_num % 1000 == 0:
                            print("{a} batches dealed, loss: {b}".format(a=train_num, b=loss_val))
                        _batch_inputs = []
                        _batch_labels = []

        embedding_result = session.run(embeddings)
        with open(embeddings_file_name, 'w') as f:
            print(str(vocabulary_size) + ' ' + str(embedding_dim) + '\n', file=f, end='')
            for word, id in word_dict.items():
                print(word, file=f, end='')
                for j in range(embedding_dim):
                    print(' ' + str(embedding_result[id][j]), file=f, end='')
                print('\n', end='', file=f)



def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    data_file = open(data_file, 'r')
    data = json.load(data_file)
    sentence = data['sentence']
    word_dict = data['words']
    train(sentence, word_dict, num_steps, embedding_dim, embeddings_file_name)


def process_data(input_data):
    data = {'sentence': [], 'words': {}}
    nlp = spacy.load('en')
    index = 0
    with zipfile.ZipFile(input_data, "r") as zipf:
        for file in zipf.namelist():
            ## check file type
            if not ('.txt' == file[-4:]): continue
            with zipf.open(file, 'r') as f:
                for line in f:
                    line = tf.compat.as_str(line)
                    doc = nlp(line)
                    ## add words to sentence and transform them to lower cases
                    sentence = []
                    for token in doc:
                        if not token.is_alpha: continue
                        sentence.append(token.text.lower())
                    if (sentence == []): continue
                    for word in sentence:
                        if word not in data['words']:
                            if (len(data['words'])<vocabulary_size):
                                data['words'][word] = index
                                index += 1
                    data['sentence'].append(sentence)
    output_file = 'data_file.json'
    with open(output_file, 'w') as out:
        json.dump(data, out)

    return output_file  # Remove this pass line, you need to implement your code to process data here...


def Compute_topk(model_file, input_adjective, top_k):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    print('adj_synonyms[{}] = '.format(input_adjective),end = '')
    return [a for a, b in model.most_similar(positive=[input_adjective], topn=top_k)]
