from typing import List

import numpy as np
import torch
import transformers
from data_utils import Span


def get_top_valid_spans(
    context_list: List[str],
    question_list: List[str],
    inputs: transformers.tokenization_utils_base.BatchEncoding,
    outputs: transformers.tokenization_utils_base.BatchEncoding,
    offset_mapping_batch: torch.Tensor,
    n_best_size: int = 1,
    max_answer_length: int = 100,
) -> List[List[Span]]:
    """
    Inference function to get top valid spans.
    https://huggingface.co/docs/transformers/tasks/question_answering

    Args:
        context_list (List[str]): context strings in  batch.
        question_list (List[str]): question strings in batch.
        inputs (transformers.tokenization_utils_base.BatchEncoding): inputs.
        outputs (transformers.tokenization_utils_base.BatchEncoding): outputs.
        offset_mapping_batch (torch.Tensor): offset mapping.
        n_best_size (int): number of best spans. Defaults to 1.
        max_answer_length (int): max answer length. Defaults to 100.

    Returns:
        List[List[Span]]: top valid spans for each isntance in batch.
    """

    batch_size = len(offset_mapping_batch)

    # TODO: remove hardcode
    entity_mapper = {
        "location": "LOC",
        "person": "PER",
        "organization": "ORG",
        "miscellaneous entity": "MISC",
    }

    assert batch_size == len(context_list)
    assert batch_size == len(question_list)
    assert batch_size == len(inputs["input_ids"])
    assert batch_size == len(inputs["token_type_ids"])
    assert batch_size == len(outputs["start_logits"])
    assert batch_size == len(outputs["end_logits"])

    top_valid_spans_batch = []

    # TODO: optimize it
    for i in range(batch_size):
        context = context_list[i]

        offset_mapping = offset_mapping_batch[i].cpu().numpy()
        mask = inputs["token_type_ids"][i].bool().cpu().numpy()
        offset_mapping[~mask] = [0, 0]
        offset_mapping = [
            (span if span != [0, 0] else None) for span in offset_mapping.tolist()
        ]

        start_logits = outputs["start_logits"][i].cpu().numpy()
        end_logits = outputs["end_logits"][i].cpu().numpy()

        start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
        end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
        top_valid_spans = []

        # TODO: validate loop
        for start_index, end_index in zip(start_indexes, end_indexes):
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or offset_mapping[start_index] is None
                or offset_mapping[end_index] is None
            ):
                continue
            if (end_index < start_index) or (
                end_index - start_index + 1 > max_answer_length
            ):
                continue
            if start_index <= end_index:
                start_context_char_char, end_context_char_char = offset_mapping[
                    start_index
                ]
                span = Span(
                    token=context[start_context_char_char:end_context_char_char],
                    label=entity_mapper[  # TODO: remove hardcode
                        question_list[i].lstrip("What is the ").rstrip("?")
                    ],
                    start_context_char_pos=start_context_char_char,
                    end_context_char_pos=end_context_char_char,
                )
                top_valid_spans.append(span)
        top_valid_spans_batch.append(top_valid_spans)
    return top_valid_spans_batch
