from collections import namedtuple
from typing import List

import torch
from tqdm import tqdm

from qaner.data_utils import Span

Instance = namedtuple("Instance", ["context", "question", "answer"])


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        qa_sentences: List[str],
        qa_labels: List[List[Span]],
    ):
        super().__init__()
        self.dataset = self._prepare_dataset(
            qa_sentences=qa_sentences,
            qa_labels=qa_labels,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Instance:
        return self.dataset[idx]

    @staticmethod
    def _prepare_dataset(
        qa_sentences: List[str],
        qa_labels: List[List[Span]],
    ) -> List[Instance]:

        dataset = []

        for sentence, labels in tqdm(zip(qa_sentences, qa_labels)):
            # TODO: remove hardcode
            for label_tag, label_name in [
                ("LOC", "location"),
                ("PER", "person"),
                ("ORG", "organization"),
                ("MISC", "miscellaneous entity"),
            ]:
                question_prompt = f"What is the {label_name}?"

                answer_list = []
                for span in labels:
                    if span.label == label_tag:
                        answer_list.append(span)

                if len(answer_list) == 0:
                    instance = Instance(
                        context=sentence,
                        question=question_prompt,
                        answer=Span(
                            token="",
                            label="O",
                            start_pos=0,
                            end_pos=0,
                        ),
                    )
                    dataset.append(instance)
                else:
                    for answer in answer_list:
                        instance = Instance(
                            context=sentence,
                            question=question_prompt,
                            answer=answer,
                        )
                        dataset.append(instance)

        return dataset
