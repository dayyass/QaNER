from typing import Any, Dict

import torch
from data_utils import Instance, Span
from inference_utils import get_top_valid_spans
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from utils import set_global_seed


def predict(
    context: str,
    question: str,  # TODO: change with entity type
    model: AutoModelForQuestionAnswering,
    tokenizer: AutoTokenizer,
    tokenizer_kwargs: Dict[str, Any],
) -> Instance:
    """
    Predict function.

    Args:
        context (str): context string.
        question (str): question string.
        model (AutoModelForQuestionAnswering): trained QaNER model.
        tokenizer (AutoTokenizer): tokenizer for trained QaNER model.
        tokenizer_kwargs (Dict[str, Any]): tokenizer kwargs.

    Returns:
        Instance: predicted instance.
    """

    inputs = tokenizer([question], [context], **tokenizer_kwargs).to(model.device)
    offset_mapping_batch = inputs.pop("offset_mapping")

    with torch.no_grad():
        outputs = model(**inputs)

    spans_pred_batch_top_1 = get_top_valid_spans(
        context_list=[context],
        question_list=[question],
        inputs=inputs,
        outputs=outputs,
        offset_mapping_batch=offset_mapping_batch,
        n_best_size=1,  # TODO: remove hardcode
        max_answer_length=100,  # TODO: remove hardcode
    )

    # TODO: maybe move into get_top_valid_spans
    # TODO: maybe remove it
    for i in range(len(spans_pred_batch_top_1)):
        if not spans_pred_batch_top_1[i]:
            empty_span = Span(
                token="",
                label="O",  # TODO: maybe not "O" label
                start_context_char_pos=0,
                end_context_char_pos=0,
            )
            spans_pred_batch_top_1[i] = [empty_span]

    predicted_answer_span = spans_pred_batch_top_1[0][0]  # TODO: remove hardcode

    start_pos = predicted_answer_span.start_context_char_pos
    end_pos = predicted_answer_span.end_context_char_pos
    assert predicted_answer_span.token == context[start_pos:end_pos]

    prediction = Instance(
        context=context,
        question=question,
        answer=predicted_answer_span,
    )

    return prediction


if __name__ == "__main__":

    # TODO argparse
    # TODO: add batch inference
    context = "\" We do n't support any such recommendation because we do n't see any grounds for it , \" the Commission 's chief spokesman Nikolaus van der Pas told a news briefing ."
    question = "What is the organization?"

    set_global_seed(42)  # TODO: remove hardcode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # bert model
    model_name = "dayyass/qaner-conll-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).eval().to(device)

    # TODO: validate it
    tokenizer_kwargs = {
        "max_length": 512,
        "truncation": "only_second",
        "padding": True,
        "return_tensors": "pt",
        "return_offsets_mapping": True,
    }

    prediction = predict(
        context=context,
        question=question,
        model=model,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
    )

    print(f"\nquestion: {prediction.question}\n")
    print(f"context: {prediction.context}")
    print(f"\nanswer: {prediction.answer}\n")
