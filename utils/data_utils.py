from torch.utils.data import DataLoader, Dataset
from typing import TypedDict, Callable, Tuple
import torch
import transformers
import pandas as pd
import numpy as np
import re
import string
from pathlib import Path


class ProcessedText(TypedDict):
    ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class TextProcessor:
    """
    Class to process text data for input to a neural network model.
    """

    def __init__(
        self,
        clean_callable: Callable,
        tokenizer_name: str = "bert-base-cased",
        max_length: int = 200,
    ) -> None:
        """
        Initialize the TextProcessor.

        Args:
            clean_callable (Callable): A function to clean the text data.
            tokenizer_name (str, optional): Name of the tokenizer to use. Defaults to "bert-base-cased".
            max_length (int, optional): Maximum length of the input text sequences. Defaults to 200.
        """
        self._tokenizer = transformers.BertTokenizer.from_pretrained(tokenizer_name)
        self._clean = clean_callable
        self._max_length = max_length

    def __call__(self, text: str, labels: np.array) -> ProcessedText:
        """
        Process the input text and labels.

        Args:
            text (str): Input text data.
            labels (np.array): Labels associated with the text data.

        Returns:
            ProcessedText: Processed text and labels.
        """
        clean_text = self._clean(text)
        tokenized_text = self._tokenizer.encode_plus(
            clean_text,
            add_special_tokens=True,
            max_length=self._max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
        )
        return {
            "ids": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


class ToxicDataset(Dataset):
    """
    Dataset class for toxic comment classification.
    """

    def __init__(
        self, sentences: pd.Series, toxic_labels: pd.Series, processor: TextProcessor
    ):
        """
        Initialize the ToxicDataset.

        Args:
            sentences (pd.Series): Series containing text data.
            toxic_labels (pd.Series): Series containing labels for toxic comments.
            processor (TextProcessor): TextProcessor instance for processing text data.
        """
        self.sentences = sentences
        self.targets = toxic_labels.to_numpy()
        self.processor = processor

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.processor(self.sentences[idx], self.targets[idx])


def clean_text(text: str) -> str:
    """
    Clean the input text data.

    Args:
        text (str): Input text data.

    Returns:
        str: Cleaned text data.
    """
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)

    return text


def split_data(data: pd.DataFrame, ratio: float) -> Tuple[pd.DataFrame, ...]:
    """
    Split the input data into training and testing sets.

    Args:
        data (pd.DataFrame): Input DataFrame containing the data.
        ratio (float): Ratio of data to be used for training.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing training and testing DataFrames.
    """
    kfold = data.shape[0] // ratio
    data["kfold"] = data.index % kfold
    train = data[data["kfold"] != 0].reset_index(drop=True)
    valid = data[data["kfold"] == 0].reset_index(drop=True)
    return train, valid


def get_dataloaders(path_to_data: Path, ratio: float, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Get DataLoader instances for training and testing data.

    Args:
        path_to_data (Path): Path to the CSV file containing the data.
        ratio (float): Ratio of data to be used for training.
        batch_size (int): Batch size for DataLoader instances.

    Returns:
        Tuple[DataLoader, DataLoader]: Tuple containing training and testing DataLoader instances.
    """
    data = pd.read_csv(path_to_data)
    text_processor = TextProcessor(clean_callable=clean_text)
    train_data, test_data = split_data(data=data, ratio=ratio)
    train_dataset = ToxicDataset(
        sentences=train_data["comment_text"],
        toxic_labels=train_data[
            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        ],
        processor=text_processor,
    )
    test_dataset = ToxicDataset(
        sentences=test_data["comment_text"],
        toxic_labels=test_data[
            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        ],
        processor=text_processor,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader
