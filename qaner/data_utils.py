from collections import namedtuple
from typing import List, Tuple

from tqdm import tqdm, trange

Span = namedtuple(
    "Span", ["token", "label", "start_context_char_pos", "end_context_char_pos"]
)
Instance = namedtuple("Instance", ["context", "question", "answer"])


def read_bio_markup(
    path: str,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Read BIO-markup data.
    Labels and tokens separated on each line with tab.
    Sentences are separated by empty line.

    Data example:
    label_11    token_11
    label_12    token_12
    label_21    token_21
    label_22    token_22
    label_23    token_23
    ...

    Args:
        path (str): Path to BIO-markup data.

    Returns:
        Tuple[List[List[str]], List[List[str]]]: Tokens and labels lists.
    """

    token_seq = []
    label_seq = []
    with open(path, mode="r") as fp:
        tokens = []
        labels = []
        for line in tqdm(fp, desc="read bio-markup"):
            if line != "\n":
                label, token = line.strip().split("\t")
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
        current_char_pos = 0
        qa_sent, qa_lab = [], []
        for token, label in zip(token_seq[i], label_seq[i]):
            qa_sent.append(token)
            if label != "O":
                span = Span(
                    token=token,
                    label=label,
                    start_context_char_pos=current_char_pos,
                    end_context_char_pos=current_char_pos + len(token),
                )
                qa_lab.append(span)
            current_char_pos += len(token) + 1
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
                    start_context_char_pos=span.start_context_char_pos,
                    end_context_char_pos=span.end_context_char_pos,
                )
                qa_lab_v2.append(span_v2)
            elif span.label.startswith("I-"):
                # TODO: remove duplicates and optimize
                span_v2 = Span(  # TODO: maybe use Span as dataclass
                    token=f"{span_v2.token} {span.token}",
                    label=span_v2.label,
                    start_context_char_pos=span_v2.start_context_char_pos,
                    end_context_char_pos=span.end_context_char_pos,
                )
                qa_lab_v2[-1] = span_v2
            else:
                raise ValueError(f"Unknown label: {span.label}")
        qa_labels_v2.append(qa_lab_v2)

    return qa_sentences, qa_labels_v2
