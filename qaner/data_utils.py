from collections import namedtuple
from typing import List, Tuple

from tqdm import tqdm, trange

Span = namedtuple("Span", ["token", "label", "start_pos", "end_pos"])


def read_conll_data_format(
    path: str,
    sep: str = "\t",
    lower: bool = True,
    verbose: bool = True,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Read data in CoNNL like format.
    Tokens and labels separated on each line.
    Sentences are separated by empty line.
    Labels should already be in necessary format, e.g. IO, BIO, BILUO, ...

    Data example:
    token_11    label_11
    token_12    label_12
    token_21    label_21
    token_22    label_22
    token_23    label_23
    ...

    Args:
        path (str): Path to txt file.
        sep (str, optional): Separator. Defaults to "\t".
        lower (bool, optional): Lowercase. Defaults to True.
        verbose (bool, optional): Verbose. Defaults to True.

    Returns:
        Tuple[List[List[str]], List[List[str]]]: Tokens and labels lists.
    """

    token_seq = []
    label_seq = []
    with open(path, mode="r") as fp:
        tokens = []
        labels = []
        if verbose:
            fp = tqdm(fp, desc="read_conll")
        for line in fp:
            if line != "\n":
                token, label = line.strip().split(sep)
                if lower:
                    token = token.lower()
                tokens.append(token)
                labels.append(label)
            else:
                if len(tokens) > 0:
                    token_seq.append(tokens)
                    label_seq.append(labels)
                tokens = []
                labels = []

    return token_seq, label_seq


def prepare_sentences_and_spans(
    token_seq: List[List[str]],
    label_seq: List[List[str]],
) -> Tuple[List[str], List[List[Span]]]:
    """
    Prepare sentences and label spans.

    Args:
        token_seq (List[List[str]]): Tokens lists.
        label_seq (List[List[str]]): Labels lists.

    Raises:
        Exception: Validate BIO-markup.

    Returns:
        Tuple[List[str], List[List[Span]]]: Sentences and label spans.
    """

    qa_sentences, qa_labels = [], []

    for i in trange(len(token_seq)):
        current_pos = 0
        qa_sent, qa_lab = [], []
        for token, label in zip(token_seq[i], label_seq[i]):
            qa_sent.append(token)
            if label != "O":
                span = Span(
                    token=token,
                    label=label,
                    start_pos=current_pos,
                    end_pos=current_pos + len(token),
                )
                qa_lab.append(span)
            current_pos += len(token) + 1
        qa_sentences.append(" ".join(qa_sent))
        qa_labels.append(qa_lab)

    qa_labels_v2 = []
    for qa_lab in qa_labels:
        qa_lab_v2 = []
        for span in qa_lab:
            if span.label.startswith("B-"):
                span_v2 = Span(
                    token=span.token,
                    label=span.label.split("-")[-1],
                    start_pos=span.start_pos,
                    end_pos=span.end_pos,
                )
                qa_lab_v2.append(span_v2)
            elif span.label.startswith("I-"):
                # TODO: remove duplicates and optimize
                span_v2 = Span(
                    token=f"{span_v2.token} {span.token}",
                    label=span_v2.label,
                    start_pos=span_v2.start_pos,
                    end_pos=span.end_pos,
                )
                qa_lab_v2[-1] = span_v2
            else:
                raise ValueError(f"Unknown label: {span.label}")
        qa_labels_v2.append(qa_lab_v2)

    return qa_sentences, qa_labels_v2
