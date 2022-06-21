import numpy as np
import torch
from inference_utils import get_top_valid_spans
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


# TODO: add metrics calculation
def train(
    n_epochs: int,
    model: AutoModelForQuestionAnswering,
    tokenizer: AutoTokenizer,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    device: torch.device,
) -> None:
    """
    Training loop.

    Args:
        n_epochs (int): number of epochs to train.
        model (AutoModelForQuestionAnswering): model.
        tokenizer (AutoTokenizer): tokenizer.
        train_dataloader (torch.utils.data.DataLoader): train_dataloader.
        test_dataloader (torch.utils.data.DataLoader): test_dataloader.
        optimizer (torch.optim.Optimizer): optimizer.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
    """

    for epoch in range(n_epochs):

        print(f"Epoch [{epoch+1} / {n_epochs}]\n")

        train_epoch(
            model=model,
            tokenizer=tokenizer,
            dataloader=train_dataloader,
            optimizer=optimizer,
            writer=writer,
            device=device,
            epoch=epoch,
        )
        evaluate_epoch(
            model=model,
            tokenizer=tokenizer,
            dataloader=test_dataloader,
            writer=writer,
            device=device,
            epoch=epoch,
        )


def train_epoch(
    model: AutoModelForQuestionAnswering,
    tokenizer: AutoTokenizer,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
) -> None:
    """
    One training cycle (loop).

    Args:
        model (AutoModelForQuestionAnswering): QA model.
        tokenizer (AutoTokenizer): tokenizer.
        dataloader (torch.utils.data.DataLoader): dataloader.
        optimizer (torch.optim.Optimizer): optimizer.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
        epoch (int): number of current epochs.
    """

    model.train()

    epoch_loss = []

    for i, inputs in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="loop over train batches",
    ):
        optimizer.zero_grad()

        instances_batch = inputs.pop("instances")

        inputs = inputs.to(device)
        offset_mapping_batch = inputs.pop("offset_mapping")

        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        writer.add_scalar(
            "batch loss / train", loss.item(), epoch * len(dataloader) + i
        )

        with torch.no_grad():
            model.eval()
            outputs_inference = model(**inputs)
            top_1_spans_batch = get_top_valid_spans(  # noqa: F841
                tokenizer=tokenizer,
                inputs=inputs,
                outputs=outputs_inference,
                offset_mapping_batch=offset_mapping_batch,
                instances_batch=instances_batch,
                n_best_size=1,
                max_answer_length=100,
            )

    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")
    writer.add_scalar("loss / train", avg_loss, epoch)


def evaluate_epoch(
    model: AutoModelForQuestionAnswering,
    tokenizer: AutoTokenizer,
    dataloader: torch.utils.data.DataLoader,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
) -> None:
    """
    One evaluation cycle (loop).

    Args:
        model (AutoModelForQuestionAnswering): model.
        tokenizer (AutoTokenizer): tokenizer.
        dataloader (torch.utils.data.DataLoader): dataloader.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
        epoch (int): number of current epochs.
    """

    model.eval()

    epoch_loss = []

    with torch.no_grad():

        for i, inputs in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="loop over test batches",
        ):

            inputs = inputs.to(device)
            # offset_mapping_batch = inputs.pop("offset_mapping")  # TODO

            outputs = model(**inputs)
            loss = outputs.loss

            epoch_loss.append(loss.item())
            writer.add_scalar(
                "batch loss / test", loss.item(), epoch * len(dataloader) + i
            )

        avg_loss = np.mean(epoch_loss)
        print(f"Test loss:  {avg_loss}\n")
        writer.add_scalar("loss / test", avg_loss, epoch)
