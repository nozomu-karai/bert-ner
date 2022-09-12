import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW

from dataset import NERDataset
from model import NERModel
from evaluate import evaluate
from task import get_dict
from utils import save_prediction


def main():
    parser = ArgumentParser()
    parser.add_argument("--task", help="ner domain", type=str, choices=['ud-japanese'], default='ud-japanese')
    parser.add_argument("--train_data", help="train data path", type=str, required=True)
    parser.add_argument("--dev_data", help="dev data path", type=str, required=True) 
    parser.add_argument(
        "--output_dir", help="output directory", type=str, required=True
    )
    parser.add_argument(
        "--pretrained_model",
        default="nlp-waseda/roberta-base-japanese",
        help="pretrained BERT model path",
    )
    parser.add_argument(
        "--max_seq_len", default=256, help="max sequence length for BERT input"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lr", default=2e-5, help="learning rate")
    parser.add_argument(
        "--weight-decay",
        default=0.01,
        help="penalty to prevent the model weights from having too large values, to avoid overfitting",
    )
    parser.add_argument("--num_epochs", default=20, help="number of epochs")
    args = parser.parse_args()
    print(args, file=sys.stderr)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = 2022
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, do_lower_case = False, do_basic_tokenize=False)
    ne_dict = get_dict(args.task)

    train_dataset = NERDataset(
            args.train_data, args.max_seq_len, tokenizer, ne_dict
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    dev_dataset = NERDataset(
            args.dev_data, args.max_seq_len, tokenizer, ne_dict
    )
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    model = NERModel(args.pretrained_model, ne_dict)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    best_score = -1
    for epoch in range(args.num_epochs):
        print(f'Start epoch {epoch}...')
        model.train()
        num_correct, size, total_loss = 0, 0, 0
        train_bar = tqdm(train_dataloader)
        for i, batch in enumerate(train_bar):
            inputs = batch['input']
            labels = batch['label']
            valid = batch['valid']

            b = labels.shape[0]
            input_ids = inputs['input_ids'].to(device).view(b, -1)
            attention_mask = inputs['attention_mask'].to(device).view(b, -1)
            token_type_ids = inputs['token_type_ids'].to(device).view(b, -1)
            labels = labels.to(device)
            valid = valid.to(device)

            loss, out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, valid=valid, label=labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() 
            prediction = torch.argmax(out, dim=2)
            prediction= prediction[valid == 1]
            labels = labels[valid == 1]
            for pred, gold in zip(prediction.view(-1), labels.view(-1)):
                if gold == -1 or gold == ne_dict['O']:
                    continue
                if pred == gold:
                    num_correct += 1
                size += 1

            train_bar.set_postfix(
                {
                    "size": size,
                    "accuracy": round(num_correct / size, 3),
                    "loss": round(total_loss / (i + 1), 3),
                }
            )

        score, prediction, label = evaluate(model, dev_dataloader, ne_dict, device)
        if score > best_score:
            print('save best model!!', file=sys.stderr)
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_best.pth'))
            best_score = score

        save_prediction(dev_dataset, prediction, label, os.path.join(args.output_dir, 'dev_prediction.jsonl'))


if __name__ == '__main__':
    main()
