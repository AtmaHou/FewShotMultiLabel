import os


def rewrite_in_data(data: [([(str, str)], str)], file_path):
    with open(file_path, 'w+', encoding='utf-8') as file:

        for sentence_label_pairs in data:
            token_tag_pairs = sentence_label_pairs[0]
            sentence = ' '.join([token_tag_pair[0] for token_tag_pair in token_tag_pairs]) + '\n'
            try:
                file.write(sentence)
            except UnicodeEncodeError:
                print('Something wrong happened')
                print(sentence)
                exit(1)


def rewrite_out_data(data: [([(str, str)], str)], file_path):
    with open(file_path, 'w+', encoding='utf-8') as file:
        for sentence_label_pairs in data:
            token_tag_pairs = sentence_label_pairs[0]
            sentence = ' '.join([token_tag_pair[1] for token_tag_pair in token_tag_pairs]) + '\n'
            file.write(sentence)


def rewrite_label_data(data: [([(str, str)], str)], file_path):
    with open(file_path, 'w+', encoding='utf-8') as file:
        for sentence_label_pairs in data:
            label = sentence_label_pairs[1]
            file.write(label + '\n')


def rewrite_data(data: [([(str, str)], str)], name, part):
    try:
        os.makedirs('./data/raw_data/%s/%s/' % (name, part))
    except FileExistsError:
        pass

    rewrite_in_data(data, './data/raw_data/%s/%s/seq.in' % (name, part))
    rewrite_out_data(data, './data/raw_data/%s/%s/seq.out' % (name, part))
    rewrite_label_data(data, './data/raw_data/%s/%s/label' % (name, part))
