# QaNER
Named Entity Recognition via Question Answering

### How to use

Train:
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

Infrerence:
```
python qaner/inference.py \
--context 'EU rejects German call to boycott British lamb .' \
--question 'What is the organization?' \
--path_to_trained_model 'dayyass/qaner-conll-bert-base-uncased' \
--seed 42
```

Possible inference question for CoNLL-2003:
- What is the location? (LOC)
- What is the person? (PER)
- What is the organization? (ORG)
- What is the miscellaneous entity? (MISC)
