import string

from utils.data_writer import rewrite_data
from utils.data_loader import read_lines


def parse_lines(lines: [str]) -> [([(str, str)], str)]:
    sentences = []
    sentence = []
    for line in lines:
        if line == '\n':
            if sentence:
                sentences.append((sentence, 'Defence'))
                sentence = []
        else:
            if not line.split() or not len(line.split()) == 2:
                continue
            else:
                token = line.split()[0]
                token = token.translate(str.maketrans('', '', string.punctuation))
                if token != '':
                    token = token.lower()
                    tag = line.split()[1]
                    sentence.append((token, tag))
    return sentences

if __name__ == '__main__':
    train_lines = read_lines('./data/unformatted_data/re3d/train.txt')
    valid_lines = []
    test_lines = read_lines('./data/unformatted_data/re3d/test.txt')


    train_data = parse_lines(train_lines)
    valid_data = parse_lines(valid_lines)
    test_data = parse_lines(test_lines)

    rewrite_data(train_data, 're3d', 'train')
    rewrite_data(valid_data, 're3d', 'valid')
    rewrite_data(test_data, 're3d', 'test')
