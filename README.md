# Spam Filter based on Deep Learning and Information Retrieval

This is the capstone project for JHUISI. Deep learning models are based on LSTM and CNN, and IR is based on Elasticsearch for noise reduction.

This project aims to improve the performance of spam filters, most of which are based on statistical models. **I first introduce deep learning models for text to classification, and then innovatively introduce IR techniques for noise reduction.**

I use precision as the evaluation metric, as it can show the FP rate of the spam filter. The results of both LSTM and CNN models gets much better than the baseline model, which is Naive Bayes. CNN reaches about **98%** percent of precision in both datasets. And noise reduction module based on Elasticsearch further helps the models, with precision improved to nearly **99%**. More detailed information can be reached from: [paper link](http://weichengzhang.co/src/paper/Capstone_final_report.pdf) or [homepage](http://weichengzhang.co).

All copyright reserved to Weicheng Zhang.

## Dataset

In this project, I used two datasets: **Grumble** and **Ling-Spam**. 

* Grumble: A dataset with 5,574 labeled messages in total, with 4,827 hams and 747 spams. Each message in this dataset has relatively shorter in length, with mostly 15 to 40 words per message. Also, words in each message has strong connection with each other, which makes each message has clear sentence meaning.

* Ling-Spam: This dataset contains 2,893 labeled messages in total, with 2,412 hams and 481 spams. Unlike Grumble dataset, this dataset is much longer in length for each message, with more than 500 words per message. Weak context information.

## Models

Baseline model: Naive Bayes

Proposed models: 1. two layer LSTM + noise reduction; 2. CNN for text + noise reduction

## Run

All tests can be run using ipynb.

