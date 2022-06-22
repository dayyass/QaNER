from typing import Any, Dict, List, Tuple

import torch
import transformers
from data_utils import Instance, Span
from tqdm import tqdm


# TODO: add documentation
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        qa_sentences: List[str],
        qa_labels: List[List[Span]],
        prompt_mapper: Dict[str, str],
    ):
        super().__init__()
        self.prompt_mapper = prompt_mapper
        self.dataset = self._prepare_dataset(
            qa_sentences=qa_sentences,
            qa_labels=qa_labels,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Instance:
        return self.dataset[idx]

    def _prepare_dataset(
        self,
        qa_sentences: List[str],
        qa_labels: List[List[Span]],
    ) -> List[Instance]:

        dataset = []

        for sentence, labels in tqdm(
            zip(qa_sentences, qa_labels),
            desc="prepare_dataset",
        ):
            for label_tag, label_name in self.prompt_mapper.items():
                question_prompt = f"What is the {label_name}?"

                answer_list = []
                for span in labels:
                    if span.label == label_tag:
                        answer_list.append(span)

                if len(answer_list) == 0:
                    empty_span = Span(
                        token="",
                        label="O",
                        start_context_char_pos=0,
                        end_context_char_pos=0,
                    )
                    instance = Instance(
                        context=sentence,
                        question=question_prompt,
                        answer=empty_span,
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


# TODO: add documentation
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
        start_end_context_char_pos_list = []

        for instance in batch:
            context_list.append(instance.context)
            question_list.append(instance.question)
            start_end_context_char_pos_list.append(
                [
                    instance.answer.start_context_char_pos,
                    instance.answer.end_context_char_pos,
                ]
            )

        tokenized_batch = self.tokenizer(
            question_list, context_list, **self.tokenizer_kwargs
        )

        offset_mapping_batch = tokenized_batch["offset_mapping"].numpy().tolist()

        assert len(offset_mapping_batch) == len(start_end_context_char_pos_list)

        # convert char boudaries to token boudaries in batch
        start_token_pos_list, end_token_pos_list = [], []
        for offset_mapping, start_end_context_char_pos in zip(
            offset_mapping_batch, start_end_context_char_pos_list
        ):
            # TODO: add comment
            if start_end_context_char_pos == [0, 0]:
                start_token_pos_list.append(0)
                end_token_pos_list.append(0)
            else:
                (
                    start_token_pos,
                    end_token_pos,
                ) = self.convert_char_boudaries_to_token_boudaries(
                    offset_mapping=offset_mapping,
                    start_end_context_char_pos=start_end_context_char_pos,
                )
                start_token_pos_list.append(start_token_pos)
                end_token_pos_list.append(end_token_pos)

        tokenized_batch["start_positions"] = torch.LongTensor(start_token_pos_list)
        tokenized_batch["end_positions"] = torch.LongTensor(end_token_pos_list)

        # additional
        tokenized_batch["instances"] = batch

        return tokenized_batch

    # TODO: add tests
    @staticmethod
    def convert_char_boudaries_to_token_boudaries(
        offset_mapping: List[List[int]],
        start_end_context_char_pos: List[int],
    ) -> Tuple[int, int]:

        start_context_char_pos, end_context_char_pos = start_end_context_char_pos
        assert end_context_char_pos >= start_context_char_pos

        done = False
        special_tokens_cnt = 0
        for i, token_boudaries in enumerate(offset_mapping):

            if token_boudaries == [0, 0]:
                special_tokens_cnt += 1
                continue

            if special_tokens_cnt == 2:
                start_token_pos, end_token_pos = token_boudaries

                if start_token_pos == start_context_char_pos:
                    res_start_token_pos = i

                if end_token_pos == end_context_char_pos:
                    res_end_token_pos = i  # inclusive
                    done = True
                    break

        assert done

        return res_start_token_pos, res_end_token_pos
