# QaNER: Prompting Question Answering Models for Few-shot Named Entity Recognition
Unofficial implementation of [QaNER](https://arxiv.org/abs/2203.01543).

You can adopt this pipeline for arbitrary [BIO-markup](https://github.com/dayyass/QaNER/tree/main/data/conll2003) data.

### CoNLL-2003
Pipeline results on CoNLL-2003 dataset:
- [Metrics](https://tensorboard.dev/experiment/FEsbNJdmSd2LGVhga8Ku0Q/)
- [Trained Hugging Face model](https://huggingface.co/dayyass/qaner-conll-bert-base-uncased)

## How to use
### Training
Script for training QaNER model:
```
python qaner/train.py \
--bert_model_name 'bert-base-uncased' \
--path_to_prompt_mapper 'prompt_mapper.json' \
--path_to_train_data 'data/conll2003/train.bio' \
--path_to_test_data 'data/conll2003/test.bio' \
--path_to_save_model 'dayyass/qaner-conll-bert-base-uncased' \
--n_epochs 2 \
--batch_size 128 \
--learning_rate 1e-5 \
--seed 42 \
--log_dir 'runs/qaner'
```

Required arguments:
- **--bert_model_name** - base bert model for QaNER fine-tuning
- **--path_to_prompt_mapper** - path to prompt mapper json file
- **--path_to_train_data** - path to train data ([BIO-markup](https://github.com/dayyass/QaNER/tree/main/data/conll2003))
- **--path_to_test_data** - path to test data ([BIO-markup](https://github.com/dayyass/QaNER/tree/main/data/conll2003))
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
--path_to_prompt_mapper 'prompt_mapper.json' \
--path_to_trained_model 'dayyass/qaner-conll-bert-base-uncased' \
--n_best_size 1 \
--max_answer_length 100 \
--seed 42
```

Result:
```
question: What is the organization?

context: EU rejects German call to boycott British lamb .

answer: [Span(token='EU', label='ORG', start_context_char_pos=0, end_context_char_pos=2)]
```

Required arguments:
- **--context** - sentence to extract entities from
- **--question** - question prompt with entity name to extract (examples below)
- **--path_to_prompt_mapper** - path to prompt mapper json file
- **--path_to_trained_model** - path to trained QaNER model
- **--n_best_size** - number of best QA answers to consider

Optional arguments:
- **--max_answer_length** - entity max length to eliminate very long entities (default: 100)
- **--seed** - random seed for reproducibility (default: 42)

Possible inference questions for CoNLL-2003:
- What is the location? (LOC)
- What is the person? (PER)
- What is the organization? (ORG)
- What is the miscellaneous entity? (MISC)

### Requirements
Python >= 3.7

### Citation
```bibtex
@misc{liu2022qaner,
    title         = {QaNER: Prompting Question Answering Models for Few-shot Named Entity Recognition},
    author        = {Andy T. Liu and Wei Xiao and Henghui Zhu and Dejiao Zhang and Shang-Wen Li and Andrew Arnold},
    year          = {2022},
    eprint        = {2203.01543},
    archivePrefix = {arXiv},
    primaryClass  = {cs.LG}
}
```
