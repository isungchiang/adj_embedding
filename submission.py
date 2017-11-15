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
batch_size = 128
num_samples = 2
num_sampled = 64
learning_rate = 0.002
skip_window = 8
vocabulary_size = 10000
nlp = spacy.load('en')
data_index = 0

def build_dataset(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # i.e., one of the 'UNK' words
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def generate_batch(batch_size, num_samples, skip_window):
    global data_index
    assert batch_size % num_samples == 0
    assert num_samples <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # span is the width of the sliding window
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])  # initial buffer content = first sliding window
    data_index += span
    for i in range(batch_size // num_samples):
        context_words = [w for w in range(span) if w != skip_window]
        random.shuffle(context_words)
        words_to_use = collections.deque(context_words)  # now we obtain a random list of context words
        for j in range(num_samples):  # generate the training pairs
            batch[i * num_samples + j] = buffer[skip_window]
            context_word = words_to_use.pop()
            labels[i * num_samples + j, 0] = buffer[context_word]  # buffer[context_word] is a random context word

        # slide the window to the next position
        if data_index == len(data):
            buffer = data[:span]
            data_index = span
        else:
            buffer.append(data[
                              data_index])  # note that due to the size limit, the left most word is automatically removed from the buffer.
            data_index += 1
    # end-of-for
    data_index = (data_index + len(data) - span) % len(data)  # move data_index back by `span`
    return batch, labels

def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    data_file = open(data_file, 'r')
    inputfile = json.load(data_file)
    _data = inputfile['mydata']
    global data,count,dictionary,reversed_dictionary
    count = [['UNK', -1]]
    count.extend(collections.Counter(_data).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in _data:
        index = dictionary.get(word, 0)
        if index == 0:  # i.e., one of the 'UNK' words
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

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
                                                             num_sampled=num_sampled, num_classes=vocabulary_size))
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            # Add variable initializer.
            init = tf.global_variables_initializer()
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
    with tf.Session(graph=graph) as session:
        session.run(init)
        print('Initializing the model')
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_samples, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val
            if step % 1000 == 0:
                if step > 0:
                    average_loss /= 5000
                    print("{a} batches dealed, loss: {b}".format(a=step, b=loss_val))
                    average_loss = 0

        embedding_result = normalized_embeddings.eval()
        with open(embeddings_file_name, 'w') as f:
            print(str(vocabulary_size) + ' ' + str(embedding_dim) + '\n', file=f, end='')
            for word, id in dictionary.items():
                print(word, file=f, end='')
                for j in range(embedding_dim):
                    print(' ' + str(embedding_result[id][j]), file=f, end='')
                print('\n', end='', file=f)


def process_data(input_data):
    _data = []
    with zipfile.ZipFile(input_data, "r") as zipf:
        for file in zipf.namelist():
            ## check file type
            if not ('.txt' == file[-4:]): continue
            with zipf.open(file, 'r') as f:
                documents = tf.compat.as_str(f.read())
                doc = nlp(documents)

                for token in doc:
                    if not token.is_alpha: continue
                    if token.pos_ == 'ADJ':
                        _data.append(token.text.lower())
                    else:
                        if token.lemma_.isalpha():
                            _data.append(token.lemma_)
                        else:
                            _data.append(token.text.lower())
    datafile = {'mydata': _data}
    output_file = 'data_file.json'
    with open(output_file, 'w') as out:
        json.dump(datafile, out)
    return output_file  # Remove this pass line, you need to implement your code to process data here...


def Compute_topk(model_file, input_adjective, top_k):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    print('adj_synonyms[{}] = '.format(input_adjective), end='')
    # return [a for a, b in model.most_similar(positive=[input_adjective], topn=top_k)]
    topk_adj = []
    i = 0
    while len(topk_adj) < top_k:
        i = i + 1
        tmp = [a for a, b in model.most_similar(positive=[input_adjective], topn=top_k * i)]
        for word in tmp:
            token = nlp(word)[0]
            if token.pos_ == 'ADJ' and (not word in topk_adj):
                topk_adj.append(word)
    return topk_adj[:top_k]
