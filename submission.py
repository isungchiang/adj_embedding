## Submission.py for COMP6714-Project2
###################################################################################################################
import math
import random
import zipfile
import numpy as np
import tensorflow as tf
import spacy
import json
import gensim

# Tunable Parameters:
batch_size = 128
num_samples = 1
num_sampled = 100
learning_rate = 0.002
skip_window = 1
nlp = spacy.load('en')


def adjective_embeddings(data_file, embeddings_file_name, num_steps, embedding_dim):
    data_file = open(data_file, 'r')
    data_dict = json.load(data_file)
    # read data from json file
    document_list = data_dict['document_list']
    frequency_word = data_dict['freq']
    global adj
    adj = data_dict['adj_word']

    # -------------------------------
    # ----------build data set-------
    # -------------------------------
    word_dictionary = {}
    index = 0
    threshold = 20
    # i set a threshold and only record the word that appear times bigger than it
    for word, freq in frequency_word.items():
        if freq < threshold: continue
        word_dictionary[word] = index
        index += 1
    # set the word that not appear a lot to UNK
    for _doc in document_list:
        for i in range(len(_doc)):
            if frequency_word[_doc[i]] < threshold:
                _doc[i] = 'UNK'
    vocabulary_size = len(word_dictionary)

    # -------------------------------
    # ----------build graph----------
    # -------------------------------
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

    # -------------------------------
    # ----------training-------------
    # -------------------------------
    with tf.Session(graph=graph) as session:
        session.run(init)
        print('Initializing the model')
        _batch_inputs = []
        _batch_labels = []
        doc_num = 0
        train_num = 0
        random.shuffle(document_list)
        while (True):
            current_doc = document_list[doc_num]
            # update sentence id
            doc_num = (doc_num + 1) % len(document_list)
            # generate batch
            for i in range(len(current_doc)):
                if current_doc[i] == 'UNK':
                    continue
                # get the window position
                win_start = max(0, i - skip_window)
                win_end = min(len(current_doc) - 1, i + skip_window + 1)
                # go random sampled
                context_words = [w for w in range(win_start, win_end) if w != i]
                if len(context_words) == 0: continue
                true_length = min(len(context_words), num_samples)
                random_shuffed_context = random.sample(context_words, true_length)
                for index in random_shuffed_context:
                    if current_doc[index] == 'UNK': continue
                    input_id = word_dictionary.get(current_doc[i])
                    label_id = word_dictionary.get(current_doc[index])
                    _batch_inputs.append(input_id)
                    _batch_labels.append(label_id)
                    # reach a batch capacity
                    if (len(_batch_inputs) == batch_size):
                        _batch_inputs = np.array(_batch_inputs, dtype=np.int32)
                        _batch_labels = np.array(_batch_labels, dtype=np.int32)
                        _batch_labels = np.reshape(_batch_labels, [batch_size, 1])
                        feed_dict = {train_inputs: _batch_inputs, train_labels: _batch_labels}
                        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                        train_num = train_num + 1
                        if train_num % 5000 == 0:
                            print("{} batches over, loss: {}".format(train_num, loss_val))
                        _batch_inputs = []
                        _batch_labels = []
                        # training end
                        # write out adj embedding file
                        if (train_num >= num_steps):
                            embedding_result = session.run(normalized_embeddings)
                            with open(embeddings_file_name, 'w') as f:
                                print(str(vocabulary_size) + ' ' + str(embedding_dim) + '\n', file=f, end='')
                                for word, id in word_dictionary.items():
                                    print(word, file=f, end='')
                                    for j in range(0, embedding_dim):
                                        print(' ' + str(embedding_result[id][j]), file=f, end='')
                                    print('\n', end='', file=f)
                            return


def process_data(input_data):
    # data will record each separate doc and word frequency. And build a adj word list
    data = {'document_list': [],  'freq': {}, 'adj_word': []}
    size = 0
    with zipfile.ZipFile(input_data, "r") as zipf:
        for file in zipf.namelist():
            ## check file type
            if not ('.txt' == file[-4:]): continue
            with zipf.open(file, 'r') as f:
                _doc = tf.compat.as_str(f.read())
                doc = nlp(_doc)
                document = []
                for token in doc:
                    if not token.is_alpha:continue
                    if (token.pos_ == 'ADJ'):
                        if token.text.lower() not in data['adj_word']:
                            data['adj_word'].append(token.text.lower())
                        document.append(token.text.lower())
                    else:
                        if token.lemma_.isalpha():
                            document.append(token.lemma_)
                        else:
                            document.append(token.text.lower())
                if (document == []): continue
                for word in document:
                    if word not in data['freq']:
                        data['freq'][word] = 1
                        size += 1
                    else:
                        data['freq'][word] += 1
                data['document_list'].append(document)
    output_file = 'data_file.json'
    with open(output_file, 'w') as out:
        json.dump(data, out)
    return output_file


def Compute_topk(model_file, input_adjective, top_k):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    # get the topk similar word and return
    topk_adj = []
    i = 0
    while len(topk_adj)<top_k:
       i = i+1
       tmp = [a for a, b in model.most_similar(positive=[input_adjective], topn=top_k*i)]
       for word in tmp:
           if not word in adj: continue
           if not word in topk_adj:
               topk_adj.append(word)
    return topk_adj[:top_k]
