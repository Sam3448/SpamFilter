
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import os
from distutils.version import LooseVersion
import warnings
from collections import Counter
import csv
import sys


# In[2]:


import numpy as np
np.set_printoptions(threshold = 1e6)


# In[22]:

def execute(dataset, trainingSection):

    trainDataDir = '/Users/SamZhang/Documents/Capstone/dataset/' + dataset + '/train'
    posTrain = trainDataDir + '/spam/' + str(trainingSection) + '/' + dataset + '_train.spam'
    negTrain = trainDataDir + '/ham/' + str(trainingSection) + '/' + dataset + '_train.ham'

    testDataDir = '/Users/SamZhang/Documents/Capstone/dataset/' + dataset + '/test'
    posTest = testDataDir + '/spam/' + str(trainingSection) + '/' + dataset + '_test.spam'
    negTest = testDataDir + '/ham/' + str(trainingSection) + '/' + dataset + '_test.ham'

    saveDataPath = '/Users/SamZhang/Documents/Capstone/Models/runs/lstmmodel/'+ dataset + '/' + str(trainingSection) + '/'


    # In[4]:


    lstm_size = 256
    lstm_layers = 2
    batch_size = 128
    learning_rate = 0.001
    drop_out = 0.5
    epochs = 50
    embed_size = 128 
    evaluate_every = 100
    sequence_len = 200
    split_frac = 0.8


    # # Processing data

    # In[5]:


    def getTextLabel(posFilePath, negFilePath):
        import fileinput
        from string import punctuation

        labels = []
        texts = []
        for line in fileinput.input(posFilePath):
            line = line.lower()
            line = ''.join([c for c in line if c not in punctuation])
            if len(line) > 0:
                texts.append(line)
                labels.append(1)

        for line in fileinput.input(negFilePath):
            line = line.lower()
            line = ''.join([c for c in line if c not in punctuation])
            if len(line) > 0:
                texts.append(line)
                labels.append(0)
        labels = np.array(labels)
        
        return texts, labels


    # In[6]:


    def getVocabToInt(texts):
        wordSet = ' '.join(texts).split()
        counts = Counter(wordSet)
        vocab = sorted(counts, key=counts.get, reverse = True)

        vocab_to_int = {word: ii for ii, word in enumerate(vocab)}
        vocab_size = len(vocab_to_int)
        return vocab_to_int


    # In[7]:


    def getFeatures(texts, labels, vocab_to_int):
        texts_in_int = []
        for line in texts:
            texts_in_int.append([vocab_to_int[word] for word in line.split()])

        text_lens = Counter([len(x) for x in texts_in_int])
        non_zero_idx = [ii for ii, texts in enumerate(texts_in_int) if len(texts) != 0] # all data that len > 0

        texts_in_int = [texts_in_int[ii] for  ii in non_zero_idx] #all sentences
        labels = np.array([labels[ii] for ii in non_zero_idx]) #0 for ham, 1 for spam
        features = np.zeros((len(texts_in_int), sequence_len), dtype=int)
        for i, row in enumerate(texts_in_int):
             features[i, -len(row):] = np.array(row)[:sequence_len]
        
        return features


    # In[8]:


    import random
    def shuffle(texts, labels):
        curSize = len(texts)
        for i in range(curSize):
            randIndex = random.randint(0, curSize - 1)
            tempText = texts[randIndex]
            texts[randIndex] = texts[i]
            texts[i] = tempText
            
            tempLabel = labels[randIndex]
            labels[randIndex] = labels[i]
            labels[i] = tempLabel


    # In[9]:


    texts_train, labels_train = getTextLabel(posTrain, negTrain)
    shuffle(texts_train, labels_train)
    texts_test, labels_test = getTextLabel(posTest, negTest)

    texts = texts_train + texts_test
    vocab_to_int = getVocabToInt(texts)
    vocab_size = len(vocab_to_int)

    features_train = getFeatures(texts_train, labels_train, vocab_to_int)


    # # Seperate Training, Validation, Test set

    # In[10]:


    split_idx = int(len(features_train)*split_frac)
    train_x, val_x = features_train[:split_idx], features_train[split_idx:]
    train_y, val_y = labels_train[:split_idx], labels_train[split_idx:]

    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape), 
          "\nValidation set: \t{}".format(val_x.shape))


    # # Building LSTM 

    # In[11]:


    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():
        inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
        labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')


    # In[12]:


    # Size of the embedding vectors (number of units in the embedding layer)
    with graph.as_default():
        embedding = tf.Variable(tf.random_uniform((vocab_size, embed_size), -1, 1)) #generate random number from [-1, 1]
        embed = tf.nn.embedding_lookup(embedding, inputs_)


    # In[13]:


    #building cell
    with graph.as_default():
        #define lstm cell
        def lstm_cell():
            cell = tf.contrib.rnn.LSTMCell(lstm_size, 
                                           initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=2),
                                           state_is_tuple = True,
                                          reuse=tf.get_variable_scope().reuse)
            drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = keep_prob)
            return drop
        
        stack_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layers)])
        
        initial_state = state = stack_cells.zero_state(batch_size, tf.float32)


    # In[14]:


    with graph.as_default():
        outputs, final_state = tf.nn.dynamic_rnn(stack_cells, embed, initial_state=initial_state)
        predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
        
        cost = tf.losses.mean_squared_error(labels_, predictions)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


    # In[15]:


    with graph.as_default():
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    # In[16]:


    def get_batches(x, y, batch_size=100):
        n_batches = len(x)//batch_size
        x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
        for ii in range(0, len(x), batch_size):
            yield x[ii:ii+batch_size], y[ii:ii+batch_size]


    # In[17]:


    def get_batches_test(text, x, y, batch_size=100):
        n_batches = len(x)//batch_size
        text, x, y = text[:n_batches*batch_size], x[:n_batches*batch_size], y[:n_batches*batch_size]
        for ii in range(0, len(x), batch_size):
            yield text[ii:ii+batch_size], x[ii:ii+batch_size], y[ii:ii+batch_size]


    # In[18]:


    with graph.as_default():
        saver = tf.train.Saver()


    # In[18]:


    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer()) 
        iteration = 1
        for e in range(epochs):
            state = sess.run(initial_state)
            
            for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob: drop_out,
                        initial_state: state}
                loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
                
                if iteration%5==0:
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Train loss: {:.3f}".format(loss))

                if iteration%evaluate_every ==0:
                    val_acc = []
                    val_state = sess.run(stack_cells.zero_state(batch_size, tf.float32))
                    for x, y in get_batches(val_x, val_y, batch_size):
                        feed = {inputs_: x,
                                labels_: y[:, None],
                                keep_prob: 1,
                                initial_state: val_state}
                        batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                        val_acc.append(batch_acc)
                    print("Val acc: {:.3f}".format(np.mean(val_acc)))
                iteration +=1
        saver.save(sess, saveDataPath + "sentiment.ckpt")


    # # Testing model

    # In[19]:


    features_test = getFeatures(texts_test, labels_test, vocab_to_int)
    print(features_test[0])


    # In[23]:


    test_acc = []
    title = np.column_stack(('text', 'prediction', 'label'))
    out_path = saveDataPath + 'prediction.csv'
    print(saveDataPath)

    with open(out_path, 'w') as f:
        csv.writer(f).writerows(title)
        
        with tf.Session(graph=graph) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(saveDataPath))
            test_state = sess.run(stack_cells.zero_state(batch_size, tf.float32))
            for ii, (text, x, y) in enumerate(get_batches_test(texts_test, features_test, labels_test, batch_size), 1):
                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob: 1,
                        initial_state: test_state}
                batch_acc, test_state, batch_cor = sess.run([accuracy, final_state, correct_pred], feed_dict=feed)
                test_acc.append(batch_acc)
                predict_label = []
                for i in range(len(y)):
                    predict_label.append(y[i] if batch_cor[i] == True else 1 - y[i])
                csv.writer(f).writerows(np.column_stack((np.array(text), predict_label, y)))
            print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


    # # Evaluation and Noise Reduction

    # In[24]:


    import ES_interface as esi

    esi.metric(dataset + '_' + str(trainingSection), saveDataPath + 'prediction.csv')


execute(str(sys.argv[1]), sys.argv[2])

