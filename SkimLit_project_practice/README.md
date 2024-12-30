# NLP Project 01: SkimLit PubMed sentence classifier

## Objective

This project is to classify sentences of an abstract into five categories of `OBJECTIVE`, `METHODS`, `RESULTS`, `BACKGROUND`, and `CONCLUSIONS`. The data are from `PubMed` (see [notebook 00](https://github.com/ZYWZong/ML_Practice_Projects/blob/e60a659556b3f231576d4f5c81e0fd0e491ba57e/SkimLit_project_practice/SkimLit_data_preprocessing_and_baseline_model_00.ipynb) for details about the dataset). The goal is to build a neural network model that beats the best baseline model (built using classic statistical learning methods) on `accuracy` while maintaining at least $70$% on `precision`, `recall`, and `F-1` scores.

## Results and conclusion

We performed experiments on CNN, RNN (LSTM and GRU), and transfomer architectures. Out of these experiments, two models (`model_CNN_E1` and `model_LSTM`) have emerged as out best models given our limited computation budget. Our best model is `model_LSTM`, which achieves an accuracy of $84.51\%$ on the development data while maintaining more than $80\%$ on all other scores. We show the scores of our models illustratively below:

![pictures/image.png](https://github.com/ZYWZong/ML_Practice_Projects/blob/188da79bbd0e4a33c56bd5d794f26b6b506737b7/SkimLit_project_practice/SkimLit_results_raw/Result_scores.png)

The detailed socres are:

| Metric (%)  | Random Forest 15 (baseline) | Model CNN E1 | Model LSTM |
| :-------- | :-------: | :-------: | :-------: |
| Accuracy  | 76.10     | 78.57     | 84.51     |
| Precision | 75.48     | 78.27     | 84.96     |
| Recall    | 76.10     | 78.57     | 84.51     |
| F-1       | 75.31     | 78.24     | 84.19     |

## Experimentation process



## Acknowledgment

This project is inspired by a similar project from *TensorFlow for Deep Learning Bootcamp* taught by Daniel Bourke on Udemy.

