import string

from utils.data_loader import read_lines
from utils.data_writer import rewrite_data


def parse_lines(lines: [str]) -> [([(str, str)], str)]:
    sentences = []
    sentence = []

    for line in lines:
        if 'http' in line or 'gt' in line:
            continue
        if line == '\t\n' or line == '\n':
            if sentence:
                sentences.append((sentence, 'SocialMedia'))
                sentence = []
        else:
            token = line.split()[0]
            token = token.translate(str.maketrans('', '', string.punctuation))
            if token != '':
                tag = line.split()[-1]
                token = token.lower()
                sentence.append((token, tag))

    return sentences

if __name__ == '__main__':
    train_lines = read_lines('./data/unformatted_data/WNUT17/wnut17train.conll')
    valid_lines = read_lines('./data/unformatted_data/WNUT17/emerging.dev.conll')
    test_lines = read_lines('./data/unformatted_data/WNUT17/emerging.test.annotated')

    train_data = parse_lines(train_lines)
    valid_data = parse_lines(valid_lines)
    test_data = parse_lines(test_lines)

    rewrite_data(train_data, 'wnut17', 'train')
    rewrite_data(valid_data, 'wnut17', 'valid')
    rewrite_data(test_data, 'wnut17', 'test')
