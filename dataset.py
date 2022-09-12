import torch

from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, path, max_seq_len, tokenizer, ne_dict):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.ne_dict = ne_dict
        self.inputs, self.labels, self.valid = self._load(path)

    def _load(self, path):
        inputs, labels, valids = [], [], []
        with open(path, 'r') as f:
            for line in f:
                input, label, valid = self._get_features(line)
                inputs.append(input)
                labels.append(label)
                valids.append(valid)

        return inputs, labels, valids

    def _get_features(self, line):
        tokens = line.split()
        sbw_text, text = [], []
        sbw_text.append(self.tokenizer.cls_token)
        label = []
        valid = []
        label.append(-1)
        valid.append(0)
        for token in tokens:
            parts = token.rsplit('/', 2)
            text.append(parts[0])
            sbw = self.tokenizer.tokenize(parts[0])
            sbw_text += sbw
            for m in range(len(sbw)):
                if m == 0:
                    label.append(self.ne_dict[parts[-1]])
                    valid.append(1)
                else:
                    label.append(-1)
                    valid.append(0)
        sbw_text.append(self.tokenizer.sep_token)

        tensor_text = self.tokenizer(' '.join(text), padding='max_length', max_length=self.max_seq_len, return_tensors='pt')
        label += [-1] * (self.max_seq_len - len(label))
        valid += [0] * (self.max_seq_len - len(valid))

        return tensor_text, label, valid

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], torch.tensor(self.labels[idx]), torch.tensor(self.valid[idx])
