import torch
from data_utils import prepare_sentences_and_spans, read_conll_data_format
from dataset import Collator, Dataset
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from utils import set_global_seed

if __name__ == "__main__":

    # argparse
    # TODO: fix it
    # from arg_parse import get_train_args
    # args = get_train_args()

    # reproducibility
    set_global_seed(42)  # TODO: remove hardcode

    # bert model
    model_name = "bert-base-uncased"  # TODO: remove hardcode

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # TODO: validate it
    tokenizer_kwargs = {
        "max_length": 512,
        "truncation": "only_second",
        "padding": True,
        "return_tensors": "pt",
        # "return_offsets_mapping": True,
    }

    # dataset and dataloader
    # TODO: remove hardcode
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

    # TODO: remove hardcode
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collator,
    )

    inputs = next(iter(dataloader))
    outputs = model(**inputs)

    print(outputs.loss)
