import pandas as pd
import re
import ES

def metric(docIndex, predictionFilePath):
    table = pd.read_csv(predictionFilePath)

    text = [x for x in table['text']]
    label = [x for x in table['label']]
    prediction = [x for x in table['prediction']]
    topK = 10 #10 for sms
    max_query_len = 3000 #chars

    def getFourParams(prediction, label):
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(len(label)):
            TP += 1 if prediction[i] == 1 and label[i] == 1 else 0
            FP += 1 if prediction[i] == 1 and label[i] == 0 else 0
            TN += 1 if prediction[i] == 0 and label[i] == 0 else 0
            FN += 1 if prediction[i] == 0 and label[i] == 1 else 0
        return TP, FP, TN, FN


    def removeMarks(text):
        reg = re.compile(r'\'|\\|\[|\(|\)|\?|,|!')
        text = reg.sub('', text)
        return text


    random = 1e-3 # in case of error "divide by zero"
    def getMetrics(TP, FP, TN, FN):
        accuracy = (TP + TN) * 1.0 / (TP + TN + FP + FN + random)
        precision = TP * 1.0 / (TP + FP + random)
        recall = TP * 1.0 / (TP + FN + random)
        f1 = 2.0 * TP / (2 * TP + FP + FN + random)
        print('accuracy : %.3f \nprecision : %.3f \nrecall : %.3f \nf1 : %.3f \n'%(accuracy, precision, recall, f1))


    #ham = 0, spam = 1
    TP, FP, TN, FN = getFourParams(prediction, label)
    PIndex = []
    size = len(text)

    for i in range(size):
        if prediction[i] == 1:
            PIndex.append(i)

    print(TP, FP, TN, FN)            
    getMetrics(TP, FP, TN, FN)


    for i in PIndex:
        curText = text[i].strip('\n')
        curText = removeMarks(curText)
        if len(curText) > max_query_len:
            curText = curText[:max_query_len]
        
        response = ES.search(curText, docIndex, topK)
        res = ES.KNN(response, 'spam', 'ham')
        # print(res)
        # print(curText + '\n')
        if res == 'ham':
            if prediction[i] == 1 and label[i] == 1: 
                TP -= 1
                FN += 1
            elif prediction[i] == 1 and label[i] == 0:
                FP -= 1
                TN += 1
        elif res == 'spam':
            if prediction[i] == 0 and label[i] == 1: 
                TP += 1
                FN -= 1
            elif prediction[i] == 0 and label[i] == 0:
                FP += 1
                TN -= 1

    print(TP, FP, TN, FN)  
    getMetrics(TP, FP, TN, FN) #remember to run metric on original dataset again before run this continuously!!

