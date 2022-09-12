ner_label = {
        'ud-japanese': ['O', 'B-PERSON', 'B-DATE', 'I-DATE', 'B-QUANTITY', 'I-QUANTITY', 'B-FAC', 'I-FAC', 'B-ORDINAL', 'I-ORDINAL', 'B-ORG', 'I-ORG', 'B-NORP', 'I-NORP', 'B-GPE', 'B-LOC', 'B-PRODUCT', 'I-PRODUCT', 'I-PERSON', 'B-TITLE_AFFIX', 'I-LOC', 'B-TIME', 'I-TIME', 'I-GPE', 'B-MONEY', 'I-MONEY', 'B-PERCENT', 'I-PERCENT', 'B-EVENT', 'I-EVENT', 'I-TITLE_AFFIX', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-MOVEMENT', 'I-MOVEMENT', 'B-LANGUAGE', 'B-LAW', 'I-LAW', 'I-LANGUAGE', 'B-CARDINAL', 'I-CARDINAL', 'B-PET_NAME', 'I-PET_NAME', 'B-PHONE', 'I-PHONE'],
        }


def get_dict(task):
    ne_list = ner_label[task]
    ne_dict = {}
    for i, ne in enumerate(ne_list):
        ne_dict[ne] = i

    return ne_dict
