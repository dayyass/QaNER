from typing import Dict, List

from data_utils import Span


# TODO: add metrics over label types
def compute_metrics(
    spans_true_batch: List[Span],
    spans_pred_batch_top_1: List[List[Span]],
) -> Dict[str, float]:
    """
    Compute NER metrics.

    Args:
        spans_true_batch (List[Span]): targets.
        spans_pred_batch_top_1 (np.ndarray): predictions.

    Returns:
        Dict[str, float]: metrics.
    """

    empty_span = Span(
        token="",
        label="O",
        start_context_char_pos=0,
        end_context_char_pos=0,
    )

    accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    accuracy_denominator, precision_denominator, recall_denominator = 0, 0, 0

    for span_true, span_pred in zip(spans_true_batch, spans_pred_batch_top_1):
        span_pred = span_pred[0]

        # TODO: maybe remove condition
        if (span_true == empty_span) and (span_pred == empty_span):
            continue

        accuracy_denominator += 1
        if span_true == span_pred:
            accuracy += 1

        # precision
        # TODO: validate
        if span_pred != empty_span:
            precision_denominator += 1
            if span_true == span_pred:
                precision += 1

        # recall
        # TODO: validate
        if span_true != empty_span:
            recall_denominator += 1
            if span_true == span_pred:
                recall += 1

    if accuracy_denominator != 0:
        accuracy /= accuracy_denominator
    if precision_denominator != 0:
        precision /= precision_denominator
    if recall_denominator != 0:
        recall /= recall_denominator
    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return metrics
