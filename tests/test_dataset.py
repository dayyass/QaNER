import unittest

from qaner.data_utils import prepare_sentences_and_spans, read_conll_data_format
from qaner.dataset import Dataset, Instance, Span

token_seq, label_seq = read_conll_data_format(
    path="data/conll2003/train.txt",
    sep=" ",
    lower=False,
    verbose=True,
)

qa_sentences, qa_labels = prepare_sentences_and_spans(
    token_seq=token_seq,
    label_seq=label_seq,
)

dataset = Dataset(
    qa_sentences=qa_sentences,
    qa_labels=qa_labels,
)


class TestDataset(unittest.TestCase):
    def test_idx_0(self):
        instance_true = Instance(
            context=" ".join(token_seq[0]),
            question="What is the location?",
            answer=Span(
                token="",
                label="O",
                start_pos=0,
                end_pos=0,
            ),
        )

        instance_pred = dataset[0]
        self.assertTupleEqual(instance_true, instance_pred)

        instance_true = Instance(
            context=" ".join(token_seq[0]),
            question="What is the organization?",
            answer=Span(
                token="EU",
                label="ORG",
                start_pos=0,
                end_pos=2,
            ),
        )

        instance_pred = dataset[2]
        self.assertTupleEqual(instance_true, instance_pred)

        instance_true = Instance(
            context=" ".join(token_seq[0]),
            question="What is the miscellaneous entity?",
            answer=Span(
                token="German",
                label="MISC",
                start_pos=11,
                end_pos=17,
            ),
        )

        instance_pred = dataset[3]
        self.assertTupleEqual(instance_true, instance_pred)

    def test_idx_3(self):
        instance_true = Instance(
            context=" ".join(token_seq[3]),
            question="What is the organization?",
            answer=Span(
                token="European Commission",
                label="ORG",
                start_pos=4,
                end_pos=23,
            ),
        )

        instance_pred = dataset[15]
        self.assertTupleEqual(instance_true, instance_pred)

    def test_idx_5(self):
        instance_true = Instance(
            context=" ".join(token_seq[5]),
            question="What is the person?",
            answer=Span(
                token="Nikolaus van der Pas",
                label="PER",
                start_pos=123,
                end_pos=143,
            ),
        )

        instance_pred = dataset[24]
        self.assertTupleEqual(instance_true, instance_pred)
