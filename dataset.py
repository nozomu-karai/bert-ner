import json

import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, path, max_seq_len, tokenizer, ne_dict):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.ne_dict = ne_dict
        self.inputs, self.labels, self.valid, self.text = self._load(path)

    def _load(self, path):
        inputs, labels, valids, texts = [], [], [], []
        with open(path, 'r') as f:
            for line in f:
                input, label, valid, text = self._get_features(line)
                inputs.append(input)
                labels.append(label)
                valids.append(valid)
                texts.append(text)

        return inputs, labels, valids, texts

    def _get_features(self, line):
        res = json.loads(line)
        tokens = res['text']
        gold = res['label']
        sbw_text, text = [], []
        sbw_text.append(self.tokenizer.cls_token)
        label = []
        valid = []
        label.append(self.ne_dict['O'])
        valid.append(0)
        for i in range(len(tokens)):
            text.append(tokens[i])
            sbw = self.tokenizer.tokenize(tokens[i])
            sbw_text += sbw
            for m in range(len(sbw)):
                if m == 0:
                    label.append(self.ne_dict[gold[i]])
                    valid.append(1)
                else:
                    if gold[i] == 'O':
                        label.append(self.ne_dict[gold[i]])
                    else:
                        label.append(self.ne_dict['I'+gold[i][1:]])
                    valid.append(0)
        sbw_text.append(self.tokenizer.sep_token)
        label.append(self.ne_dict['O'])

        tensor_text = self.tokenizer(' '.join(text), padding='max_length', max_length=self.max_seq_len, return_tensors='pt')
        label += [-1] * (self.max_seq_len - len(label))
        valid += [0] * (self.max_seq_len - len(valid))

        return tensor_text, label, valid, text

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
                'input': self.inputs[idx],
                'label': torch.tensor(self.labels[idx]),
                'valid': torch.tensor(self.valid[idx]),
                }  
