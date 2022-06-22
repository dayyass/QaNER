import torch
from arg_parse import get_train_args
from data_utils import prepare_sentences_and_spans, read_conll_data_format
from dataset import Collator, Dataset
from torch.utils.tensorboard import SummaryWriter
from train_utils import train
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from utils import set_global_seed

if __name__ == "__main__":

    # argparse
    args = get_train_args()

    set_global_seed(args.seed)
    writer = SummaryWriter(log_dir=args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # bert model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(args.bert_model_name).to(
        device
    )

    # TODO: validate it
    tokenizer_kwargs = {
        "max_length": 512,
        "truncation": "only_second",
        "padding": True,
        "return_tensors": "pt",
        "return_offsets_mapping": True,
    }

    # dataset and dataloader (train)
    # TODO: add more general read_data function
    # TODO: add valid data
    train_token_seq, train_label_seq = read_conll_data_format(
        path=args.path_to_train_data,
        sep=" ",  # TODO: remove hardcode
        lower=False,  # TODO: remove hardcode
        verbose=True,  # TODO: remove hardcode
    )

    train_qa_sentences, train_qa_labels = prepare_sentences_and_spans(
        token_seq=train_token_seq,
        label_seq=train_label_seq,
    )

    train_dataset = Dataset(
        qa_sentences=train_qa_sentences,
        qa_labels=train_qa_labels,
    )

    # dataset and dataloader (test)
    test_token_seq, test_label_seq = read_conll_data_format(
        path=args.path_to_test_data,
        sep=" ",  # TODO: remove hardcode
        lower=False,  # TODO: remove hardcode
        verbose=True,  # TODO: remove hardcode
    )

    test_qa_sentences, test_qa_labels = prepare_sentences_and_spans(
        token_seq=test_token_seq,
        label_seq=test_label_seq,
    )

    test_dataset = Dataset(
        qa_sentences=test_qa_sentences,
        qa_labels=test_qa_labels,
    )

    collator = Collator(
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # train / eval
    # TODO: change to AdamW
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    train(
        n_epochs=args.n_epochs,
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        writer=writer,
        device=device,
    )

    model.save_pretrained(args.path_to_save_model)
    tokenizer.save_pretrained(args.path_to_save_model)
