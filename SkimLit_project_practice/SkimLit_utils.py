"""
This file contains helper functions for the practice project SkimLit

Acknowledgment: 
  Parts of the functions are adopted from "TensorFlow for Deep Learning Bootcamp" taught by Daniel Bourke on Udemy.
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

# Perform evaluations on prediction results
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
  accuracy = accuracy_score(y_true, y_pred) * 100
  precision, recall, f1_score, _ = precision_recall_fscore_support(y_true,y_pred, average = "weighted")
  return {"Metric": ["accuracy", "precision", "recall", "F1"],
          model_name: [accuracy, precision, recall, f1_score]}
