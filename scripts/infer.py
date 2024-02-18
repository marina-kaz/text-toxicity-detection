import torch
import transformers
from pathlib import Path
from utils.data_utils import clean_text
from utils.parameters import MAX_LENGTH
from tap import Tap
from typing import Dict, Any, Union


class InferenceCLI(Tap):
    """
    Command-line interface for model inference.
    """

    text: str
    """Text to infer"""

    model: Path
    """Path to checkpoint"""

    device: str
    """Device to put calculations on"""


def load_model(path: Path, device: str) -> transformers.PreTrainedModel:
    """
    Load the trained model from the checkpoint.

    Args:
        path (Path): Path to the model checkpoint.
        device (str): Device to load the model on.

    Returns:
        transformers.PreTrainedModel: Loaded model.
    """
    checkpoint = torch.load(path, map_location=torch.device(device))
    model = transformers.BertForSequenceClassification.from_pretrained(
        'bert-base-cased', num_labels=6
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def preprocess_text(text: str) -> Dict[str, Any]:
    """
    Preprocess the input text for inference.

    Args:
        text (str): Input text.

    Returns:
        Dict[str, Any]: Preprocessed text including token IDs, attention mask, etc.
    """
    text = clean_text(text)
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        pad_to_max_length=True,
        truncation=True,
        return_attention_mask=True,
    )


def infer_model(model: transformers.PreTrainedModel, text: Dict[str, Any], device: str) -> torch.Tensor:
    """
    Perform inference using the loaded model.

    Args:
        model (transformers.PreTrainedModel): Loaded model.
        text (Dict[str, Any]): Preprocessed text including token IDs, attention mask, etc.
        device (str): Device to perform inference on.

    Returns:
        torch.Tensor: Predictions from the model.
    """
    ids = torch.tensor(text["input_ids"]).unsqueeze(0).to(device)
    mask = torch.tensor(text["attention_mask"]).unsqueeze(0).to(device)
    output = model(ids, mask)["logits"].squeeze(-1).to(torch.float32)
    return torch.where(torch.sigmoid(output) > 0.5, 1, 0)


def parse_predictions(text: str, predictions: torch.Tensor) -> None:
    """
    Parse and print the predictions.

    Args:
        text (str): Original input text.
        predictions (torch.Tensor): Predictions from the model.
    """
    print(f"Text: {text}")
    for prediction, category in zip(predictions,
                                    "toxic severe_toxic obscene threat insult identity_hate".split()):
        print(f"{category}: {bool(int(prediction))}")


if __name__ == "__main__":
    args = InferenceCLI(underscores_to_dashes=True).parse_args()
    model = load_model(args.model, args.device)
    text = preprocess_text(args.text)
    predictions = infer_model(model, text, args.device)
    parse_predictions(args.text, predictions)
