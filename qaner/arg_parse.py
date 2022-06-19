from argparse import ArgumentParser, Namespace


def get_train_args() -> Namespace:
    """
    Training Argument Parser.

    Returns:
        Namespace: Parsed arguments.
    """

    parser = ArgumentParser()

    parser.add_argument(
        "--path_to_train_data",
        type=str,
        required=True,
        help="path to train data",
    )
    parser.add_argument(
        "--path_to_save_model",
        type=str,
        required=True,
        help="path to save model",
    )

    args = parser.parse_args()
    return args
