## BERT-NER
This is simple implementation of NER using BERT.

## News
- 2022-10-15 Update: Train with using all subword

## Usage
### Requirement
- Python 3.7
- pytorch
- Transformers

### Data preparation
Data format is json.

Need tokenized text 'text' and crresponding labels 'label'.

In detail, see `ud_data` directory.

### Training
Add label candidates list to `task.py`.

Add task choices in `train.py`
```
python train.py --train_data ud_data/ja_gsd-ud-train.ne.jsonl --dev_data ud_data/ja_gsd-ud-dev.ne.jsonl \
	--output_dir result/ud \
	--pretrained_model cl-tohoku/bert-base-japanese 
```

### Test
```
python test.py --saved_dir result/ud --test_data ud_data/ja_gsd-ud-test.ne.jsonl
```
