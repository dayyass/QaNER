from typing import Any, Dict, List

import torch
import transformers
from data_utils import Instance, Span
from tqdm import tqdm


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

        for sentence, labels in tqdm(
            zip(qa_sentences, qa_labels),
            desc="prepare_dataset",
        ):
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


class Collator:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        tokenizer_kwargs: Dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def __call__(
        self,
        batch: List[Instance],
    ) -> transformers.tokenization_utils_base.BatchEncoding:

        context_list = []
        question_list = []
        start_pos_list, end_pos_list = [], []

        for instance in batch:
            context_list.append(instance.context)
            question_list.append(instance.question)
            start_pos_list.append(instance.answer.start_pos)
            end_pos_list.append(instance.answer.end_pos)

        tokenized_batch = self.tokenizer(
            question_list, context_list, **self.tokenizer_kwargs
        )
        tokenized_batch["start_positions"] = torch.LongTensor(start_pos_list)
        tokenized_batch["end_positions"] = torch.LongTensor(end_pos_list)

        # additional
        tokenized_batch["instances"] = batch

        return tokenized_batch
