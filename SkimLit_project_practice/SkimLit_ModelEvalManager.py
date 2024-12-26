"""
This file contains:
  class ModelEvalManager

Usage:
  Add and update model evaluation metrics
"""

from github import Github
import json

class ModelEvalManager:
  def __init__(self, github_token, repo_name, filepath):
        """
        Argument:
            github_token (str): personal access token
            repo_name (str): name of the github repo
            filepath (str): path to file
        """
        self.github = Github(github_token)
        self.repo = self.github.get_repo(repo_name)
        self.file_path = filepath
        self.scores = {"Metric": ["Accuracy", "Precision", "Recall", "F1-score"]}

    def add(self, model_name, results):
        """
        Argument:
            model_name (str): name of the model
            results (list): list of scores
        """
        self.scores[model_name] = results

    def save(self, commit_message="Update model scores"):
        try:
            file_content = self.repo.get_contents(self.filepath)
            self.repo.update_file(
                path=self.filepath,
                message=commit_message,
                content=json.dumps(self.scores, indent=4),
                sha=file_content.sha,
            )
            print("File updated successfully on GitHub.")
        except Exception as e:
            print(f"Error updating file: {e}")

    def load(self):
        try:
            file_content = self.repo.get_contents(self.filepath)
            self.scores = json.loads(file_content.decoded_content.decode())
            return self.scores
        except Exception as e:
            print(f"Error loading file: {e}")
