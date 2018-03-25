# Spam Filter based on Deep Learning and Information Retrieval

This is the capstone project for JHUISI. Deep learning models are based on LSTM and CNN, and IR is based on Elasticsearch for noise reduction.

数据集：在./dataset/路径下，有sms和ling两种。分别分成了train/test,其中又分别有ham/spam.格式完全一样。运行各个模型的时候，只要改dataset参数为sms或ling即可运行。 其中1-10是训练集比例，分别代表每10个raw data就又1-10个数据被分为training data, (10 - training)分成testing data.

分类模型：./Models/路径下，其中Naive_Bayesian是baseline方法，LSTM和CNN分别可以针对训练集进行模型训练，并且可以对test进行测试，测试结果均输出到./Models/runs/xxxmodel/number/路径下的prediction.csv文件中，两个csv格式完全相同。

降噪模型：我重新把降噪模型抽象成了一个api,cnn和lstm的notebook最后的evaluation and noise reduction就可以直接输出降噪前和降噪后的结果。

其他文件都是工具类文件，不用管。

有一些路径我是写死了的，你改一下就能跑了。

SMS.txt和lingspam_public是raw data, 你应该用不着。dataset我都已经parse好了，直接用就行。

还需要安装Elasticsearch，否则没法降噪。

