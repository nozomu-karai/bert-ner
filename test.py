import os
import sys
from argparse import ArgumentParser
import time
import json
import warnings
warnings.simplefilter('ignore')

from transformers import AutoTokenizer
from transformers import logging
logging.set_verbosity_warning()
logging.set_verbosity_error()
import torch
from torch.utils.data import DataLoader

from dataset import NERDataset
from model import NERModel
from evaluate import evaluate
from task import get_dict
from utils import save_prediction



def main():
    parser = ArgumentParser()
    parser.add_argument("--test_data", help="train data path", type=str, required=True)
    parser.add_argument(
        "--saved_dir", help="saved directory", type=str, required=True
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    args = parser.parse_args()
    with open(os.path.join(args.saved_dir, 'config.json'), 'r') as f:
        config_json = json.load(f)
    args.task = config_json['task']
    args.pretrained_model = config_json['pretrained_model']
    args.max_seq_len = config_json['max_seq_len']
    print(args, file=sys.stderr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = 2022
    torch.manual_seed(seed)

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, do_lower_case = False, do_basic_tokenize=False)
    ne_dict = get_dict(args.task)
    test_dataset = NERDataset(
            args.test_data, args.max_seq_len, tokenizer, ne_dict
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = NERModel(args.pretrained_model, test_dataset.ne_dict)
    state_dict = torch.load(os.path.join(args.saved_dir, 'model_best.pth'))
    model.load_state_dict(state_dict)
    model = model.to(device)

    _, prediction, label = evaluate(model, test_dataloader, ne_dict, device)
    elapsed = time.time() - start
    print(f'elapsed: {elapsed} [sec]')
    save_prediction(test_dataset, prediction, label, os.path.join(args.saved_dir, 'test_prediction.jsonl'))



if __name__ == '__main__':
    main()
