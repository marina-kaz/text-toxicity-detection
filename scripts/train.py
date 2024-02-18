from tap import Tap
from pathlib import Path
from utils.data_utils import get_dataloaders
from utils.training_utils import Trainer, get_optimizer
import transformers
import torch

from utils.parameters import TRAIN_TEST_RATIO, BATCH_SIZE, N_EPOCH


class TrainingCLI(Tap):

    path_to_data: Path
    """Path to training data"""

    model: str
    """Model name on HF"""

    device: str
    """Device to put calculations on"""

    ckpt_path: Path
    """Path to saved model"""


if __name__ == "__main__":
    args = TrainingCLI(underscores_to_dashes=True).parse_args()

    train_dataloader, test_dataloader = get_dataloaders(
        args.path_to_data, ratio=TRAIN_TEST_RATIO, batch_size=BATCH_SIZE
    )
    model = transformers.BertForSequenceClassification.from_pretrained(
        args.model, num_labels=6
    )
    optimizer = get_optimizer(model.parameters())
    train_steps = int((len(train_dataloader) * N_EPOCH) / BATCH_SIZE)
    num_steps = int(train_steps * 0.1)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_steps, train_steps
    )
    trainer = Trainer(
        model,
        criterion=torch.nn.BCEWithLogitsLoss(),
        device=args.device,
        ckpt_path=args.ckpt_path,
    )

    trainer.train(
        train_dataloader=train_dataloader,
        valid_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler=optimizer,
        epochs=N_EPOCH,
    )
