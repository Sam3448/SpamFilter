
# coding: utf-8

# In[1]:


import TrainingSetsUtil as tsutil
import numpy as np
import sys

def execute(dataset, trainingSection):

    print(dataset, trainingSection)
    
    trainDataDir = '/Users/SamZhang/Documents/Capstone/dataset/' + dataset + '/train'
    posTrain = trainDataDir + '/spam/' + str(trainingSection) + '/'
    negTrain = trainDataDir + '/ham/' + str(trainingSection) + '/'


    # In[16]:


    ham_training_set, spam_training_set = tsutil.init(negTrain, posTrain)


    # # Classify

    # In[17]:


    testDataDir = '/Users/SamZhang/Documents/Capstone/dataset/' + dataset + '/test'
    posTest = testDataDir + '/spam/' + str(trainingSection) + '/' + dataset + '_test.spam'
    negTest = testDataDir + '/ham/' + str(trainingSection) + '/' + dataset + '_test.ham'


    # In[18]:


    def classify(message, training_set, prior = 0.5, c = 3.7e-4):
        msg_terms = tsutil.get_words(message)
        msg_probability = 1
        
        for term in msg_terms:        
            if term in training_set:
                msg_probability *= training_set[term]
            else:
                msg_probability *= c

        return msg_probability * prior


    # In[19]:


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


    # In[20]:


    random = 1e-3 # in case of error "divide by zero"
    def getMetrics(TP, FP, TN, FN):
        accuracy = (TP + TN) * 1.0 / (TP + TN + FP + FN + random)
        precision = TP * 1.0 / (TP + FP + random)
        recall = TP * 1.0 / (TP + FN + random)
        f1 = 2.0 * TP / (2 * TP + FP + FN + random)
        print('accuracy : %.3f \nprecision : %.3f \nrecall : %.3f \nf1 : %.3f \n'%(accuracy, precision, recall, f1))


    # In[21]:


    texts, labels = getTextLabel(posTest, negTest)
    len(texts)


    # In[22]:


    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(texts)):
        curText = texts[i]
        curLabel = labels[i]
        prediction = 0
        
    #     print(curText, curLabel)
        spam_probability = classify(curText, spam_training_set, 0.2)
        ham_probability = classify(curText, ham_training_set, 0.8)
        if spam_probability > ham_probability:
            prediction = 1
            
        if prediction == 0 and curLabel == 0:
            TN += 1
        elif prediction == 0 and curLabel == 1:
            FN += 1
        elif prediction == 1 and curLabel == 1:
            TP += 1
        elif prediction == 1 and curLabel == 0:
            FP += 1
            
    print(TP, FP, TN, FN)   
    getMetrics(TP, FP, TN, FN)

execute(str(sys.argv[1]), sys.argv[2])