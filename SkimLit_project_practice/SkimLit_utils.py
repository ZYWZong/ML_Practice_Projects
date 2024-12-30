"""
This file contains helper functions for the practice project SkimLit
"""

# Preprocess PubMed dataset
import os

def read_lines(filepath):
    """
    A wrapper function for reading lines of a file.

    Argument: 
        filepath (str): Path to a file.
    Return:
        list: A list of strings, each string representing a line in the file.
    """
    with open(filepath, "r") as file:
        return file.readlines()


def preprocess_data_with_line_numbers(filepath):
    """
    Processes a file to create structured data with line numbers and total lines.

    Argument:
        filepath (str): Path to the file.
    Return:
        list: A list of dictionaries where each dictionary contains:
            - 'label': The label of the line.
            - 'text': The text content of the line, converted to lowercase.
            - 'line_number': The line's number within its abstract.
            - 'total_lines': The total number of lines in the abstract.
    """
    # Use the read_lines function to read the file
    input_lines = read_lines(filepath)
    abstract_lines = ""  
    abstract_preprocessed = [] 

    # Process each line from the input
    for line in input_lines:
        if line.startswith("###"):
            abstract_id = line
            abstract_lines = ""
        elif line.isspace():
            abstract_split = abstract_lines.splitlines()
            for abstract_line_number, abstract_line in enumerate(abstract_split):
                label_split = abstract_line.split("\t")
                line_data = {
                    "label": label_split[0],
                    "text": label_split[1].lower(),
                    "line_number": abstract_line_number,
                    "total_lines": len(abstract_split) - 1
                }
                abstract_preprocessed.append(line_data)
        else:
            abstract_lines += line

    return abstract_preprocessed


import pandas as pd

def SkimLit_preprocess_master(data_dir):
  """
  Master wrapper function for data preprocessing for SkimLit project
  
  Argument:
    data_dir (str): path to data
  Return:
    a tuple containing the train, dev, and test data
  """
  train_samples = preprocess_data_with_line_numbers(data_dir + "train.txt")
  dev_samples = preprocess_data_with_line_numbers(data_dir + "dev.txt")
  test_samples = preprocess_data_with_line_numbers(data_dir + "test.txt")

  train_df = pd.DataFrame(train_samples)
  dev_df = pd.DataFrame(dev_samples)
  test_df = pd.DataFrame(test_samples)
  
  return train_df, dev_df, test_df
  

from sklearn.preprocessing import LabelEncoder

def SkimLit_preprocess_EncodedLabels(train_df,dev_df,test_df):
  label_encoder = LabelEncoder()
  train_labels_encoded = label_encoder.fit_transform(train_df["label"].to_numpy())
  dev_labels_encoded = label_encoder.transform(dev_df["label"].to_numpy())
  test_labels_encoded = label_encoder.transform(test_df["label"].to_numpy())
  return {"train_label":train_labels_encoded, "dev_label":dev_labels_encoded, "test_label":test_labels_encoded}

from sklearn.preprocessing import OneHotEncoder

def SkimLit_preprocess_OneHot_NN(train_df, dev_df, test_df):
    """
    Generate one hot encdoing for the text labels and convert the text sentences into list for
    builtin preprocessing functions in tensorflow (e.g. batching).
    
    Argument:
      train_df: pandas dataframe for training data
      dev_df: pandas dataframe for development data
      test_df: pandas dataframe for test data
    Return:
      a tuple containing the train, dev, and test sentences and their labels
    """
    train_sentences = train_df["text"].tolist()
    dev_sentences = dev_df["text"].tolist()
    test_sentences = test_df["text"].tolist()
    
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    train_labels_one_hot = one_hot_encoder.fit_transform(train_df["label"].to_numpy().reshape(-1, 1))
    dev_labels_one_hot = one_hot_encoder.transform(dev_df["label"].to_numpy().reshape(-1, 1))
    test_labels_one_hot = one_hot_encoder.transform(test_df["label"].to_numpy().reshape(-1, 1))

    return {"train_text": train_sentences, "dev_text": dev_sentences, "test_text": test_sentences,
            "train_label": train_labels_one_hot, "dev_label": dev_labels_one_hot, "test_label": test_labels_one_hot
           }


import tensorflow as tf

def SkimLit_batching_data(dict_OneHot_NN):
  """
  Argument:
    dict_OneHot_NN: return of the SkimLit_preprocess_OneHot_NN function
  Return:
    a dictionary containing batched train, dev, and test data (PrefetchDataset)
  """
  train_data = tf.data.Dataset.from_tensor_slices((dict_OneHot_NN["train_text"], 
                                                   dict_OneHot_NN["train_label"]))
  dev_data = tf.data.Dataset.from_tensor_slices((dict_OneHot_NN["dev_text"], 
                                                 dict_OneHot_NN["dev_label"]))
  test_data = tf.data.Dataset.from_tensor_slices((dict_OneHot_NN["test_text"], 
                                                  dict_OneHot_NN["test_label"]))

  train_data = train_data.batch(32).prefetch(tf.data.AUTOTUNE)
  dev_data = dev_data.batch(32).prefetch(tf.data.AUTOTUNE)
  test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

  return {"train_data": train_data, "dev_data": dev_data, "test_data": test_data}



#-------------------------------------------

# Perform evaluations on prediction results
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def perform_evaluations(y_true, y_pred,model_name):
  """
  Argument:
    y_true (array): true labels
    y_pred (array): predicted labels
    model_name (str): name of the model
  Return:
    A dictionary consisting: (1) an item called "Metric" of a list of metric names
                             (2) an item called "model_name" of a list of metric values
  """
  accuracy = accuracy_score(y_true, y_pred)
  precision, recall, f1_score, _ = precision_recall_fscore_support(y_true,y_pred, average = "weighted")
  results = np.round(np.array([accuracy, precision, recall, f1_score]),4) * 100
  return {"Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
          model_name: results}


import matplotlib.pyplot as plt

def plot_loss_curves(history):
    """
    Plot the train vs validation losses.

    Argument:
        history: tensorFlow History object
    Return:
        None
    """
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    x = range(1,len(train_loss)+1)

    plt.plot(x, train_loss, label="Training Loss")
    plt.plot(x, val_loss, label="Validation Loss")
    plt.xticks(x)
    plt.title("Loss Curves")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show();


def plot_loss_and_accuracy(history):
    """
    Plot the train vs validation losses and the train vs validation accuracy

    Argument:
        history: tensorFlow History object
    Return:
        None
    """
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    x = range(1,len(train_loss)+1)
  
    plt.figure(figsize=(12, 5))
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(x,train_loss, label="training loss")
    plt.plot(x,val_loss, label="validation loss")
    plt.xticks(x)
    plt.title("Loss Curves")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(bbox_to_anchor=(1.0, 1.0))

    # plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(x, train_accuracy, label="training accuracy")
    plt.plot(x, val_accuracy, label="validation accuracy")
    plt.xticks(x)
    plt.title("Accuracy Curves")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend(bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    plt.show()
