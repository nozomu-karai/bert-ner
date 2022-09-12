import sys
import torch
import torch.nn as nn
from tqdm import tqdm 


def evaluate(model, test_data, ne_dict, device):
    model.eval()
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    num_correct, size, total_loss = 0, 0, 0
    all_prediction, all_labels = [], []
    with torch.no_grad():
        test_bar = tqdm(test_data)
        for i, batch in enumerate(test_bar):
            inputs, labels, valid = batch
            b = labels.shape[0]
            input_ids = inputs['input_ids'].to(device).view(b, -1)
            attention_mask = inputs['attention_mask'].to(device).view(b, -1)
            token_type_ids = inputs['token_type_ids'].to(device).view(b, -1)
            labels = labels.to(device)
            valid = valid.to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            out = out * valid.unsqueeze(2)

            _, _, t = out.shape
            loss = ce_loss(out.view(-1, t), labels.view(-1))
            total_loss += loss.item()

            prediction = torch.argmax(out, dim=2)
            prediction = prediction[valid == 1]
            labels = labels[valid == 1]
            for pred, gold in zip(prediction.view(-1), labels.view(-1)):
                if gold == -1 or gold == ne_dict['O']:
                    continue
                if pred == gold:
                    num_correct += 1
                size += 1

            all_prediction.extend(prediction.view(-1).tolist())
            all_labels.extend(labels.view(-1).tolist())

            test_bar.set_postfix(
                {
                    "size": size,
                    "accuracy": round(num_correct / size, 3),
                    "loss": round(total_loss / (i + 1), 3),
                }
            )

    id2ne = {v: k for k, v in ne_dict.items()}
    all_prediction = [id2ne[pred] for pred in all_prediction]
    all_labels = [id2ne[label] for label in all_labels]

    score = ent_score(all_prediction, all_labels)

    return score


def check_correct(pred_entity, gold_entity, count):
    domain = gold_entity[0].split('-')[-1]
    if not domain in count:
        count[domain] = [0, 1]
    else:
        count[domain][1] += 1

    tp, fp, tn, fn = 0, 0, 0, 0
    if pred_entity == gold_entity:
        count[domain][0] += 1
        if gold_entity == ['O']:
            return tp, fp, tn, fn
        if pred_entity[-1] == 'O':
            tn += 1
        else:
            tp += 1
    else:
        if pred_entity[-1] == 'O':
            fn +=1
        else:
            fp +=1

    return tp, fp, tn, fn


def ent_score(y_pred, y_gold):
    num_pred = count_entity(y_pred)
    num_gold = count_entity(y_gold)
    num_correct = 0
    pred_entity, gold_entity = [], []
    count = {}
    for pred, gold in zip(y_pred, y_gold):
        if gold == 'O':
            bio_tag = '-'
        else:
            gold_ne = gold.split('-')
            bio_tag = gold_ne[0]

        if bio_tag == 'I':
            pred_entity.append(pred)
            gold_entity.append(gold)
        else:
            if len(pred_entity) != 0:
                if pred_entity == gold_entity:
                    if gold_entity[-1] != 'O':
                        num_correct += 1
                _, _, _, _ = check_correct(pred_entity, gold_entity, count)
            pred_entity, gold_entity = [], []
            pred_entity.append(pred)
            gold_entity.append(gold)

    p = num_correct / num_pred if num_pred != 0 else 0
    r = num_correct / num_gold if num_gold != 0 else 0
    f = 2 * p * r /  (p + r) if p + r != 0 else 0
    print(f'p: {p}, r: {r}, f1: {f}', file=sys.stderr)
    print(f'COR: {num_correct}, PRED: {num_pred}, GOLD: {num_gold}', file=sys.stderr)

    return f


def count_entity(label):
    count = 0
    ent = []
    for l in label:
        bio_domain = l.split('-')
        if bio_domain[0] == 'I':
            ent.append(l)
        else:
            if len(ent) != 0:
                ent = []
                count += 1
            if bio_domain[0] == 'B':
                ent.append(l)
    if len(ent) != 0:
        count += 1

    return count
