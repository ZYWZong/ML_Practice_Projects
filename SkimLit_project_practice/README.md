# NLP Project 01: SkimLit PubMed sentence classifier

## Objective

This project is to classify sentences of an abstract into five categories of `OBJECTIVE`, `METHODS`, `RESULTS`, `BACKGROUND`, and `CONCLUSIONS`. The data are from `PubMed` (see [notebook 00](https://github.com/ZYWZong/ML_Practice_Projects/blob/e60a659556b3f231576d4f5c81e0fd0e491ba57e/SkimLit_project_practice/SkimLit_data_preprocessing_and_baseline_model_00.ipynb) for details about the dataset). The goal is to build a neural network model that beats the best baseline model (built using classic statistical learning methods) on `accuracy` while maintaining at least $70$% on `precision`, `recall`, and `F-1` scores on our development data.

## Results

We performed experiments on **CNN**, **RNN** (**LSTM** and **GRU**), and **transfomer** architectures. Out of these experiments, two models (`model_CNN_E1` and `model_LSTM`) have emerged as our best models within our computation budget. Our **best** model is `model_LSTM`, which achieves an accuracy of $84.51$% on the development data while maintaining more than $80$% on all other scores. We show the scores of our models illustratively below:

![pictures/image.png](https://github.com/ZYWZong/ML_Practice_Projects/blob/2a2d8e8640c5c253c2f48a8fe0be710c54e78657/SkimLit_project_practice/SkimLit_results_raw/Result_scores.png)

The exact socres are:

| Metric (%)  | Random Forest 15 (baseline) | Model CNN E1 | Model LSTM |
| :-------- | :-------: | :-------: | :-------: |
| Accuracy  | 76.10     | 78.57     | 84.51     |
| Precision | 75.48     | 78.27     | 84.96     |
| Recall    | 76.10     | 78.57     | 84.51     |
| F-1       | 75.31     | 78.24     | 84.19     |

## Experimentation process

* In [**notebook 00**](https://github.com/ZYWZong/ML_Practice_Projects/blob/e60a659556b3f231576d4f5c81e0fd0e491ba57e/SkimLit_project_practice/SkimLit_data_preprocessing_and_baseline_model_00.ipynb), we used classic statical learning methods (Naive Bayes and Random Forests) to build the best baseline model, which we found to be a 15 tree Random Forest model.

* In [**notebook 01**](https://github.com/ZYWZong/ML_Practice_Projects/blob/42e9e455dd0a2ae73c8d9d6f2beb35d2262d5319/SkimLit_project_practice/SkimLit_experiment01_token_embeddings_with_CNN_01.ipynb), we applied word tokenization to our data and built CNN models. We find our best CNN model to be `model_CNN_E1`.

* In [**notebook 02**](https://github.com/ZYWZong/ML_Practice_Projects/blob/42e9e455dd0a2ae73c8d9d6f2beb35d2262d5319/SkimLit_project_practice/SkimLit_experiment02_multiple_embeddings_LSTM_and_GRU_02.ipynb), we applied positional, character, and token embeddings to our data and use these embeddings to build an LSTM and a GRU models. We find our best RNN model to be `model_LSTM`.

* In [**notebook 03**](https://github.com/ZYWZong/ML_Practice_Projects/blob/50fc2a44c9796ece822e85d9e788b7dd5368aabf/SkimLit_project_practice/SkimLit_experiment03_transformers_03.ipynb), we attempted transformer models. However, due to limited computation budget, we were unable to obtain satisfactory scores for these models.

* In [**notebook 04**](https://github.com/ZYWZong/ML_Practice_Projects/blob/42e9e455dd0a2ae73c8d9d6f2beb35d2262d5319/SkimLit_project_practice/SkimLit_results_04.ipynb), we summarized our results.

## Acknowledgment

This project is inspired by a similar project from *TensorFlow for Deep Learning Bootcamp* taught by Daniel Bourke on Udemy.

