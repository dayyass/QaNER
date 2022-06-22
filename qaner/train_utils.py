from collections import defaultdict

import numpy as np
import torch
from data_utils import Span
from inference_utils import get_top_valid_spans
from metrics import compute_metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForQuestionAnswering


# TODO: add metrics calculation
def train(
    n_epochs: int,
    model: AutoModelForQuestionAnswering,
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
            dataloader=train_dataloader,
            optimizer=optimizer,
            writer=writer,
            device=device,
            epoch=epoch,
        )
        evaluate_epoch(
            model=model,
            dataloader=test_dataloader,
            writer=writer,
            device=device,
            epoch=epoch,
        )


def train_epoch(
    model: AutoModelForQuestionAnswering,
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
        dataloader (torch.utils.data.DataLoader): dataloader.
        optimizer (torch.optim.Optimizer): optimizer.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
        epoch (int): number of current epochs.
    """

    model.train()

    epoch_loss = []
    batch_metrics_list = defaultdict(list)

    for i, inputs in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc="loop over train batches",
    ):
        optimizer.zero_grad()

        instances_batch = inputs.pop("instances")

        context_list, question_list = [], []
        for instance in instances_batch:
            context_list.append(instance.context)
            question_list.append(instance.question)

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
            model.train()

        spans_pred_batch_top_1 = get_top_valid_spans(
            context_list=context_list,
            question_list=question_list,
            inputs=inputs,
            outputs=outputs_inference,
            offset_mapping_batch=offset_mapping_batch,
            n_best_size=1,
            max_answer_length=100,  # TODO: remove hardcode
        )

        # TODO: maybe move into get_top_valid_spans
        for i in range(len(spans_pred_batch_top_1)):
            if not spans_pred_batch_top_1[i]:
                empty_span = Span(
                    token="",
                    label="O",  # TODO: maybe not "O" label
                    start_context_char_pos=0,
                    end_context_char_pos=0,
                )
                spans_pred_batch_top_1[i] = [empty_span]

        spans_true_batch = [instance.answer for instance in instances_batch]

        batch_metrics = compute_metrics(
            spans_true_batch=spans_true_batch,
            spans_pred_batch_top_1=spans_pred_batch_top_1,
        )

        for metric_name, metric_value in batch_metrics.items():
            batch_metrics_list[metric_name].append(metric_value)
            writer.add_scalar(
                f"batch {metric_name} / train",
                metric_value,
                epoch * len(dataloader) + i,
            )

    avg_loss = np.mean(epoch_loss)
    print(f"Train loss: {avg_loss}\n")
    writer.add_scalar("loss / train", avg_loss, epoch)

    for metric_name, metric_value_list in batch_metrics_list.items():
        metric_value = np.mean(metric_value_list)
        print(f"Train {metric_name}: {metric_value}\n")
        writer.add_scalar(f"{metric_name} / train", metric_value, epoch)


# TODO: remove train_epoch code duplicates
def evaluate_epoch(
    model: AutoModelForQuestionAnswering,
    dataloader: torch.utils.data.DataLoader,
    writer: SummaryWriter,
    device: torch.device,
    epoch: int,
) -> None:
    """
    One evaluation cycle (loop).

    Args:
        model (AutoModelForQuestionAnswering): model.
        dataloader (torch.utils.data.DataLoader): dataloader.
        writer (SummaryWriter): tensorboard writer.
        device (torch.device): cpu or cuda.
        epoch (int): number of current epochs.
    """

    model.eval()

    epoch_loss = []
    batch_metrics_list = defaultdict(list)

    with torch.no_grad():

        for i, inputs in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="loop over test batches",
        ):

            instances_batch = inputs.pop("instances")

            context_list, question_list = [], []
            for instance in instances_batch:
                context_list.append(instance.context)
                question_list.append(instance.question)

            inputs = inputs.to(device)
            offset_mapping_batch = inputs.pop("offset_mapping")

            outputs = model(**inputs)
            loss = outputs.loss

            epoch_loss.append(loss.item())
            writer.add_scalar(
                "batch loss / test", loss.item(), epoch * len(dataloader) + i
            )

            spans_pred_batch_top_1 = get_top_valid_spans(
                context_list=context_list,
                question_list=question_list,
                inputs=inputs,
                outputs=outputs,
                offset_mapping_batch=offset_mapping_batch,
                n_best_size=1,  # TODO: remove hardcode
                max_answer_length=100,  # TODO: remove hardcode
            )

            # TODO: maybe move into get_top_valid_spans
            for i in range(len(spans_pred_batch_top_1)):
                if not spans_pred_batch_top_1[i]:
                    empty_span = Span(
                        token="",
                        label="O",  # TODO: maybe not "O" label
                        start_context_char_pos=0,
                        end_context_char_pos=0,
                    )
                    spans_pred_batch_top_1[i] = [empty_span]

            # TODO: change metrics calculation for inference time
            spans_true_batch = [instance.answer for instance in instances_batch]

            batch_metrics = compute_metrics(
                spans_true_batch=spans_true_batch,
                spans_pred_batch_top_1=spans_pred_batch_top_1,
            )

            for metric_name, metric_value in batch_metrics.items():
                batch_metrics_list[metric_name].append(metric_value)
                writer.add_scalar(
                    f"batch {metric_name} / test",
                    metric_value,
                    epoch * len(dataloader) + i,
                )

        avg_loss = np.mean(epoch_loss)
        print(f"Test loss:  {avg_loss}\n")
        writer.add_scalar("loss / test", avg_loss, epoch)

        for metric_name, metric_value_list in batch_metrics_list.items():
            metric_value = np.mean(metric_value_list)
            print(f"Test {metric_name}: {metric_value}\n")
            writer.add_scalar(f"{metric_name} / test", np.mean(metric_value), epoch)
