ner_label = {
        'ud-japanese': ['O', 'PERSON', 'DATE', 'QUANTITY', 'FAC', 'ORDINAL', 'ORG', 'NORP', 'GPE', 'LOC', 'PRODUCT', 'TITLE_AFFIX', 'TIME', 'MONEY', 'PERCENT', 'EVENT', 'WORK_OF_ART', 'MOVEMENT', 'LANGUAGE', 'LAW', 'CARDINAL', 'PET_NAME', 'PHONE'],
        }


def get_dict(task):
    ne_list = ner_label[task]
    ne_dict = {}
    i = 0
    for ne in ne_list:
        if ne == 'O':
            ne_dict[ne] = i
        else:
            ne_ditc['B-'+ne] = i
            i += 1
            ne_dict['I-'+ne] = i
        i += 1

    return ne_dict
