import json
from typing import Any, Dict

import torch
from arg_parse import get_inference_args
from data_utils import Instance, Span
from inference_utils import get_top_valid_spans
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from utils import set_global_seed


# TODO: add batch inference
def predict(
    context: str,
    question: str,  # TODO: change with entity type
    prompt_mapper: Dict[str, str],
    model: AutoModelForQuestionAnswering,
    tokenizer: AutoTokenizer,
    tokenizer_kwargs: Dict[str, Any],
    n_best_size: int = 1,
    max_answer_length: int = 100,
) -> Instance:
    """
    Predict function.

    Args:
        context (str): context string.
        question (str): question string.
        prompt_mapper (Dict[str, str]): prompt mapper.
        model (AutoModelForQuestionAnswering): trained QaNER model.
        tokenizer (AutoTokenizer): tokenizer for trained QaNER model.
        tokenizer_kwargs (Dict[str, Any]): tokenizer kwargs.
        n_best_size (int): number of best QA answers to consider. Defaults to 1.
        max_answer_length (int): entity max length to eliminate very long entities. Defaults to 100.

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
        prompt_mapper=prompt_mapper,
        inputs=inputs,
        outputs=outputs,
        offset_mapping_batch=offset_mapping_batch,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
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

    # argparse
    args = get_inference_args()

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # bert model
    tokenizer = AutoTokenizer.from_pretrained(args.path_to_trained_model)
    model = (
        AutoModelForQuestionAnswering.from_pretrained(args.path_to_trained_model)
        .eval()
        .to(device)
    )

    # TODO: validate it
    tokenizer_kwargs = {
        "max_length": 512,
        "truncation": "only_second",
        "padding": True,
        "return_tensors": "pt",
        "return_offsets_mapping": True,
    }

    with open(args.path_to_prompt_mapper, mode="r", encoding="utf-8") as fp:
        prompt_mapper = json.load(fp)

    prediction = predict(
        context=args.context,
        question=args.question,
        prompt_mapper=prompt_mapper,
        model=model,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
    )

    print(f"\nquestion: {prediction.question}\n")
    print(f"context: {prediction.context}")
    print(f"\nanswer: {prediction.answer}\n")
