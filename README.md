# Spam Filter based on Deep Learning and Information Retrieval

This is the capstone project for JHUISI. Deep learning models are based on LSTM and CNN, and IR is based on Elasticsearch for noise reduction.

数据集：在./dataset/路径下，有sms和ling两种。分别分成了train/test,其中又分别有ham/spam.格式完全一样。运行各个模型的时候，只要改dataset参数为sms或ling即可运行。

分类模型：./Models/路径下，其中Naive_Bayesian是baseline方法，LSTM和CNN分别可以针对训练集进行模型训练，并且可以对test进行测试，测试结果均输出到./Models/runs/xxxmodel/路径下的prediction.csv文件中，两个csv格式完全相同。

降噪模型：两个prediction.csv文件都可以被ES_interface读取，然后运行可以分别产生降噪前后的结果。注意：在运行第二次结果前，要保证再运行一次第一个函数，否则会出现计数错误。

其他文件都是工具类文件，不用管。

有一些路径我是写死了的，你改一下就能跑了。

SMS.txt和lingspam_public是raw data, 你应该用不着。others那个文件夹是个java工程，里面是我写的parser.可以改变train/test的比例。你看看能不能运行（用个java ide）. 如果不行的话再说。路径也是写死的，你自己直接改就行，很好改。
