import sys
import unittest
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BatchEncoding

sys.path.append("qaner")  # TODO: fix it

from data_utils import prepare_sentences_and_spans, read_conll_data_format  # noqa: E402
from dataset import Collator, Dataset, Instance, Span  # noqa: E402
from utils import set_global_seed  # noqa: E402


def validate_spans(
    qa_sentences: List[str],
    qa_labels: List[List[Span]],
) -> bool:

    for sentence, labels in tqdm(
        zip(qa_sentences, qa_labels),
        desc="validate_spans",
    ):
        for span in labels:
            token = span.token
            start_context_char_pos = span.start_context_char_pos
            end_context_char_pos = span.end_context_char_pos
            assert token == sentence[start_context_char_pos:end_context_char_pos]
    return True


# reproducibility
set_global_seed(42)


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_kwargs = {  # TODO: validate it
    "max_length": 512,
    "truncation": "only_second",
    "padding": True,
    "return_tensors": "pt",
    "return_offsets_mapping": True,
}


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

collator = Collator(
    tokenizer=tokenizer,
    tokenizer_kwargs=tokenizer_kwargs,
)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collator,
)


class TestDataset(unittest.TestCase):
    def test_idx_0(self):
        instance_true = Instance(
            context=" ".join(token_seq[0]),
            question="What is the location?",
            answer=Span(
                token="",
                label="O",
                start_context_char_pos=0,
                end_context_char_pos=0,
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
                start_context_char_pos=0,
                end_context_char_pos=2,
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
                start_context_char_pos=11,
                end_context_char_pos=17,
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
                start_context_char_pos=4,
                end_context_char_pos=23,
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
                start_context_char_pos=123,
                end_context_char_pos=143,
            ),
        )

        instance_pred = dataset[24]
        self.assertTupleEqual(instance_true, instance_pred)

    def test_length(self):
        self.assertEqual(len(token_seq), len(label_seq))
        self.assertEqual(len(token_seq), 14986)

        self.assertEqual(len(qa_sentences), len(qa_labels))
        self.assertEqual(len(qa_sentences), 14986)

        self.assertEqual(len(dataset), 66658)

    def test_spans(self):
        self.assertTrue(
            validate_spans(
                qa_sentences=qa_sentences,
                qa_labels=qa_labels,
            ),
        )


class TestCollator(unittest.TestCase):
    def test_batch(self):
        batch_true = BatchEncoding()
        batch_true["input_ids"] = torch.tensor(
            [
                [
                    101,
                    2054,
                    2003,
                    1996,
                    3295,
                    1029,
                    102,
                    7327,
                    19164,
                    2446,
                    2655,
                    2000,
                    17757,
                    2329,
                    12559,
                    1012,
                    102,
                ],
                [
                    101,
                    2054,
                    2003,
                    1996,
                    2711,
                    1029,
                    102,
                    7327,
                    19164,
                    2446,
                    2655,
                    2000,
                    17757,
                    2329,
                    12559,
                    1012,
                    102,
                ],
            ]
        )
        batch_true["token_type_ids"] = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        batch_true["attention_mask"] = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        batch_true["offset_mapping"] = torch.tensor(
            [
                [
                    [0, 0],
                    [0, 4],
                    [5, 7],
                    [8, 11],
                    [12, 20],
                    [20, 21],
                    [0, 0],
                    [0, 2],
                    [3, 10],
                    [11, 17],
                    [18, 22],
                    [23, 25],
                    [26, 33],
                    [34, 41],
                    [42, 46],
                    [47, 48],
                    [0, 0],
                ],
                [
                    [0, 0],
                    [0, 4],
                    [5, 7],
                    [8, 11],
                    [12, 18],
                    [18, 19],
                    [0, 0],
                    [0, 2],
                    [3, 10],
                    [11, 17],
                    [18, 22],
                    [23, 25],
                    [26, 33],
                    [34, 41],
                    [42, 46],
                    [47, 48],
                    [0, 0],
                ],
            ]
        )
        batch_true["start_positions"] = torch.tensor([0, 0])
        batch_true["end_positions"] = torch.tensor([0, 0])
        batch_true["instances"] = [
            Instance(
                context="EU rejects German call to boycott British lamb .",
                question="What is the location?",
                answer=Span(
                    token="",
                    label="O",
                    start_context_char_pos=0,
                    end_context_char_pos=0,
                ),
            ),
            Instance(
                context="EU rejects German call to boycott British lamb .",
                question="What is the person?",
                answer=Span(
                    token="",
                    label="O",
                    start_context_char_pos=0,
                    end_context_char_pos=0,
                ),
            ),
        ]

        batch_pred = next(iter(dataloader))

        for key in batch_true:
            if key == "instances":
                self.assertListEqual(batch_true[key], batch_pred[key])
            else:
                self.assertTrue(
                    torch.allclose(batch_true[key], batch_pred[key]),
                )


if __name__ == "__main__":
    unittest.main()
