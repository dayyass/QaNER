from argparse import ArgumentParser, Namespace


def get_train_args() -> Namespace:
    """
    Training Argument Parser.

    Returns:
        Namespace: Parsed arguments.
    """

    parser = ArgumentParser()

    parser.add_argument(
        "--bert_model_name",
        type=str,
        required=True,
        help="bert model name",
    )
    parser.add_argument(
        "--path_to_prompt_mapper",
        type=str,
        required=True,
        help="path to prompt mapper",
    )
    parser.add_argument(
        "--path_to_train_data",
        type=str,
        required=True,
        help="path to train data",
    )
    parser.add_argument(
        "--path_to_test_data",
        type=str,
        required=True,
        help="path to test data",
    )
    parser.add_argument(
        "--path_to_save_model",
        type=str,
        required=True,
        help="path to save model",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        required=True,
        help="number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="seed for reproducibility",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=False,
        default="runs/qaner",
        help="tensorboard log_dir",
    )

    args = parser.parse_args()
    return args


def get_inference_args() -> Namespace:
    """
    Infrerence Argument Parser.

    Returns:
        Namespace: Parsed arguments.
    """

    parser = ArgumentParser()

    parser.add_argument(
        "--context",
        type=str,
        required=True,
        help="sentence to extract entities from",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="question prompt with entity name to extract",
    )
    parser.add_argument(
        "--path_to_prompt_mapper",
        type=str,
        required=True,
        help="path to prompt mapper",
    )
    parser.add_argument(
        "--path_to_trained_model",
        type=str,
        required=True,
        help="path to trained QaNER model",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        required=True,
        help="number of best QA answers to consider",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        required=False,
        default=100,
        help="entity max length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="seed for reproducibility",
    )

    args = parser.parse_args()
    return args
