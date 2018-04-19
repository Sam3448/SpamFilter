# Spam Filter based on Deep Learning and Information Retrieval

This is the capstone project for JHUISI. Deep learning models are based on LSTM and CNN, and IR is based on Elasticsearch for noise reduction.

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

## Result

Use only precision as metric for performance evaluation. Details will be revealed later.

