ner_label = {
        'ud-japanese': []
        }

ne_dict = {'O': 0, 'B-PERSON': 1, 'B-DATE': 2, 'I-DATE': 3, 'B-QUANTITY': 4, 'I-QUANTITY': 5, 'B-FAC': 6, 'I-FAC': 7, 'B-ORDINAL': 8, 'I-ORDINAL': 9, 'B-ORG': 10, 'I-ORG': 11, 'B-NORP': 12, 'I-NORP': 13, 'B-GPE': 14, 'B-LOC': 15, 'B-PRODUCT': 16, 'I-PRODUCT': 17, 'I-PERSON': 18, 'B-TITLE_AFFIX': 19, 'I-LOC': 20, 'B-TIME': 21, 'I-TIME': 22, 'I-GPE': 23, 'B-MONEY': 24, 'I-MONEY': 25, 'B-PERCENT': 26, 'I-PERCENT': 27, 'B-EVENT': 28, 'I-EVENT': 29, 'I-TITLE_AFFIX': 30, 'B-WORK_OF_ART': 31, 'I-WORK_OF_ART': 32, 'B-MOVEMENT': 33, 'I-MOVEMENT': 34, 'B-LANGUAGE': 35, 'B-LAW': 36, 'I-LAW': 37, 'I-LANGUAGE': 38, 'B-CARDINAL': 39, 'I-CARDINAL': 40, 'B-PET_NAME': 41, 'I-PET_NAME': 42, 'B-PHONE': 43, 'I-PHONE': 44}


def get_dict(task):
    ne_list = ner_label[task]
    ne_dict = {}
    for i, ne in enumerate(ne_list):
        ne_dict[ne] = i

    return ne_dict
