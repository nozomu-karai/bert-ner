import os
import sys
from argparse import ArgumentParser

from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

from dataset import NERDataset
from model import NERModel
from evaluate import evaluate
from task import get_dict



def main():
    parser = ArgumentParser()
    parser.add_argument("--task", help="ner domain", type=str, choices=['ud-japanese'], default='ud-japanese')
    parser.add_argument("--test_data", help="train data path", type=str, required=True)
    parser.add_argument(
        "--saved_dir", help="saved directory", type=str, required=True
    )
    parser.add_argument(
        "--pretrained-model",
        default="nlp-waseda/roberta-base-japanese",
        help="pretrained BERT model path",
    )
    parser.add_argument(
        "--max-seq-len", default=256, help="max sequence length for BERT input"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    args = parser.parse_args()
    print(args, file=sys.stderr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = 2022
    torch.manual_seed(seed)

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

    score = evaluate(model, test_dataloader, ne_dict, device)


if __name__ == '__main__':
    main()
