import json


def save_prediction(dataset, prediction, label, path):
    text = dataset.text
    n_word = 0
    f = open(path, 'a')
    for sentence in text:
        result = {}
        result['text'] = sentence
        result['prediction'] = prediction[n_word:n_word+len(sentence)]
        result['label'] = label[n_word:n_word+len(sentence)]
        n_word += len(sentence)
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

    f.close()
