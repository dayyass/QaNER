# QaNER: Prompting Question Answering Models for Few-shot Named Entity Recognition

Unofficial implementation of [QaNER](https://arxiv.org/abs/2203.01543).

## How to use

### Training

Script for training QaNER model:
```
python qaner/train.py \
--bert_model_name 'bert-base-uncased' \
--path_to_train_data 'data/conll2003/train.txt' \
--path_to_test_data 'data/conll2003/test.txt' \
--path_to_save_model 'dayyass/qaner-conll-bert-base-uncased' \
--n_epochs 2 \
--batch_size 128 \
--learning_rate 1e-5 \
--seed 42 \
--log_dir 'runs/qaner'
```

Required arguments:
- **--bert_model_name** - base bert model for QaNER fine-tuning
- **--path_to_train_data** - path to train data ([CoNLL-2003 like format](https://github.com/dayyass/QaNER/tree/main/data/conll2003))
- **--path_to_test_data** - path to test data ([CoNLL-2003-like format](https://github.com/dayyass/QaNER/tree/main/data/conll2003))
- **--path_to_save_model** - path to save trained QaNER model
- **--n_epochs** - number of epochs to fine-tune
- **--batch_size** - batch size
- **--learning_rate** - learning rate

Optional arguments:
- **--seed** - random seed for reproducibility (default: 42)
- **--log_dir** - tensorboard log_dir (default: 'runs/qaner')

### Infrerence

Script for inference trained QaNER model:
```
python qaner/inference.py \
--context 'EU rejects German call to boycott British lamb .' \
--question 'What is the organization?' \
--path_to_trained_model 'dayyass/qaner-conll-bert-base-uncased' \
--seed 42
```

Required arguments:
- **--context** - sentence to extract entities from
- **--question** - question prompt with entity name to extract (examples below)
- **--path_to_trained_model** - path to trained QaNER model

Optional arguments:
- **--seed** - random seed for reproducibility (default: 42)

Possible inference questions for CoNLL-2003:
- What is the location? (LOC)
- What is the person? (PER)
- What is the organization? (ORG)
- What is the miscellaneous entity? (MISC)

### Requirements
Python >= 3.7
