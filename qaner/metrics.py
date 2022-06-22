from typing import Dict, List

import numpy as np
from data_utils import Span


# TODO: add metrics over label types
def compute_metrics(
    spans_true_batch: List[Span],
    spans_pred_batch_top_1: List[List[Span]],
    prompt_mapper: Dict[str, str],
) -> Dict[str, float]:
    """
    Compute NER metrics.

    Args:
        spans_true_batch (List[Span]): targets.
        spans_pred_batch_top_1 (np.ndarray): predictions.
        prompt_mapper (Dict[str, str]): prompt mapper.

    Returns:
        Dict[str, float]: metrics.
    """

    metrics = {}

    entity_mapper = {"O": 0}
    for entity_tag in prompt_mapper:
        entity_mapper[entity_tag] = len(entity_mapper)

    ner_confusion_matrix = np.zeros((len(entity_mapper), len(entity_mapper)))
    confusion_matrix_true_denominator = np.zeros(len(entity_mapper))
    confusion_matrix_pred_denominator = np.zeros(len(entity_mapper))

    for span_true, span_pred in zip(spans_true_batch, spans_pred_batch_top_1):
        span_pred = span_pred[0]

        i = entity_mapper[span_true.label]
        j = entity_mapper[span_pred.label]  # type: ignore

        confusion_matrix_true_denominator[i] += 1
        confusion_matrix_pred_denominator[j] += 1

        if span_true == span_pred:
            ner_confusion_matrix[i, j] += 1

    assert (
        confusion_matrix_true_denominator.sum()
        == confusion_matrix_pred_denominator.sum()
    )

    ner_confusion_matrix_diag = np.diag(ner_confusion_matrix)

    # TODO: hide RuntimeWarning
    accuracy = np.nan_to_num(
        ner_confusion_matrix_diag.sum() / confusion_matrix_true_denominator.sum()
    )
    precision_per_entity_type = np.nan_to_num(
        ner_confusion_matrix_diag / confusion_matrix_pred_denominator
    )
    recall_per_entity_type = np.nan_to_num(
        ner_confusion_matrix_diag / confusion_matrix_true_denominator
    )
    f1_per_entity_type = np.nan_to_num(
        2
        * precision_per_entity_type
        * recall_per_entity_type
        / (precision_per_entity_type + recall_per_entity_type)
    )

    metrics["accuracy"] = accuracy

    for label_tag, idx in entity_mapper.items():
        metrics[f"precision_tag_{label_tag}"] = precision_per_entity_type[idx]
        metrics[f"recall_tag_{label_tag}"] = recall_per_entity_type[idx]
        metrics[f"f1_tag_{label_tag}"] = f1_per_entity_type[idx]

    # TODO: add micro average

    # macro average
    metrics["precision_macro"] = precision_per_entity_type.mean()
    metrics["recall_macro"] = recall_per_entity_type.mean()
    metrics["f1_macro"] = f1_per_entity_type.mean()

    # weighted average
    metrics["precision_weighted"] = np.average(
        precision_per_entity_type,
        weights=confusion_matrix_true_denominator,
    )
    metrics["recall_weighted"] = np.average(
        recall_per_entity_type,
        weights=confusion_matrix_true_denominator,
    )
    metrics["f1_weighted"] = np.average(
        f1_per_entity_type,
        weights=confusion_matrix_true_denominator,
    )

    return metrics
