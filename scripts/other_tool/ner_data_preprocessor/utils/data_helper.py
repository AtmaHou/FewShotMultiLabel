# coding:utf-8
import json, os
from collections import Counter
from itertools import chain
import shutil


def preprocess_one_sample(tokenizer, word_lst, label_lst, do_indexing=False):
    sentence = ' '.join(word_lst)
    tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(sentence)
    indexed_data = tokenizer.convert_tokens_to_ids(tokenized_text) if do_indexing else None
    segment_id = [0 for i in range(len(tokenized_text))]
    word_piece_mark = [int((len(w) > 2) and w[0] == '#' and w[1] == '#') for w in tokenized_text]
    word_piece_label = []
    label_idx = 0
    for ind, mark in enumerate(word_piece_mark):
        wp_label = None
        if mark == 0:
            wp_label = label_lst[label_idx]
            label_idx += 1
        elif mark == 1:
            wp_label = word_piece_label[-1]
            wp_label = wp_label.replace('B-', 'I-') if 'B-' in wp_label else wp_label
        if not wp_label:
            raise RuntimeError('Empty label')
        word_piece_label.append(wp_label)
    if not (len(word_piece_label) == len(tokenized_text) == len(word_piece_mark)):
        raise RuntimeError('{}{}{}{}{}{}{}{}{}{}{}'.format(
            'Failed to generate new sequence labels:{}{}{}'.format(
                len(word_piece_label), len(tokenized_text), len(word_piece_mark)),
            '\nword_piece_label', word_piece_label,
            '\ntokenized_text', tokenized_text,
            '\nword_piece_mark', word_piece_mark,
            '\nseq_in', word_lst,
            '\nseq_out', label_lst)
         )
    return indexed_data, word_piece_mark, segment_id, tokenized_text, word_piece_label


def get_word_piece_labeled_data(tokenizer, seq_ins, seq_outs, do_indexing=False):
    """
    expend slot filling label to word piece style
    output1: new_seq_ins
    output2: new_seq_outs
    """
    indexed_datas, word_piece_marks, segment_ids, tokenized_texts, word_piece_labels = [], [], [], [], []
    for seq_in, seq_out in zip(seq_ins, seq_outs):
        # TODO: Override the wordpiece tokenizer to add
        preprocessed_data = preprocess_one_sample(tokenizer, seq_in, seq_out, do_indexing)
        indexed_data, word_piece_mark, segment_id, tokenized_text, word_piece_label = preprocessed_data
        if do_indexing:
            indexed_datas.append(indexed_data)
        word_piece_marks.append(word_piece_mark)
        segment_ids.append(segment_id)
        tokenized_texts.append(tokenized_text)
        word_piece_labels.append(word_piece_label)
    if do_indexing:
        return indexed_datas, word_piece_marks, segment_ids, tokenized_texts, word_piece_labels
    else:
        return word_piece_marks, segment_ids, tokenized_texts, word_piece_labels


def load_conll_data(path):
    data_set = []
    label_set = []
    with open(path, 'r') as reader:
        lines = reader.read().split('\n\n')
        for sent in lines:
            words = []
            labels = []
            word_label_pairs = sent.split('\n')
            for word_label_pair in word_label_pairs:
                if word_label_pair.strip():
                    word, label = word_label_pair.strip().split()
                    words.append(word)
                    labels.append(label)
            data_set.append(words)
            label_set.append(labels)
    return data_set, label_set


def load_my_json_data(path):
    with open(path, 'r') as reader:
        raw_data = json.load(reader)
        data_set = raw_data['seq_ins']
        label_set = raw_data['seq_outs']
    return data_set, label_set


def load_one_batch(batch_data, word_piece):
    if not word_piece:
        support_data_set = batch_data['support']['seq_ins']
        support_label_set = batch_data['support']['seq_outs']
        test_data_set = batch_data['batch']['seq_ins']
        test_label_set = batch_data['batch']['seq_outs']
    else:  # load data split with word piece
        support_data_set = batch_data['support']['tokenized_texts']
        support_label_set = batch_data['support']['word_piece_labels']
        test_data_set = batch_data['batch']['tokenized_texts']
        test_label_set = batch_data['batch']['word_piece_labels']  # split label for word piece
    return support_data_set, support_label_set, test_data_set, test_label_set


def load_few_shot_data(path, batch_id=0, style='test', word_piece=False):
    """
    the target file must contain only one domain & one batch!
    input:
        path: file path
        batch_id=0, the batch num to load, set <0  to load all batches
        style='test': return style, choice: ['test', 'pretrain']
        word_piece=False
    """
    with open(path, 'r') as reader:
        raw_data = json.load(reader)
    if style == 'test':
        if len(raw_data) > 1:
            print(json.dumps(raw_data, indent=2))
            raise RuntimeError('Wrong raw data content')
        domain_n, domain = [(d_n, d) for d_n, d in raw_data.items()][0]  # test style: only load the first domain
        if batch_id > 0:  # only load one batch
            batch_data = domain[batch_id]  # only load the 'batch_id' th batch
            support_data_set, support_label_set, test_data_set, test_label_set = load_one_batch(batch_data, word_piece)
            return support_data_set, support_label_set, test_data_set, test_label_set
        else:  # load and return all batches
            all_batches = []
            for batch_data in domain:
                items = load_one_batch(batch_data,word_piece)
                all_batches.append(items)
            return all_batches
    elif style == 'all':
        all_data = []
        for domain_n, domain in raw_data.items():
            all_batches = []
            for batch_data in domain:
                items = load_one_batch(batch_data, word_piece)
                all_batches.append(items)
            all_data.append([domain_n, all_batches])
        return all_data
    else:
        raise ValueError('Wrong style choice:', style, 'expecting one of follow:', ['test', 'pretrain'])


def load_target_domain_data(opt):
    """ load train & test data from target domain """
    all_data_lst = []
    all_few_shot_batches = load_few_shot_data(path=opt.test_path, batch_id=-1, word_piece=opt.word_piece)
    for few_shot_batch in all_few_shot_batches:
        train_x, train_y, test_x, test_y = few_shot_batch
        dev_x, dev_y = train_x, train_y  # no dev data on target domain
        all_data_lst.append([train_x, train_y, dev_x, dev_y, test_x, test_y])
    return all_data_lst

#
# def load_target_domain_data(train_path, dev_path, test_path, word_piece, batch_id=0):
#     train_data, train_label, test_data, test_label = load_few_shot_data(path=test_path, batch_id=batch_id, style='test', word_piece=word_piece)
#     dev_data, dev_label = train_data, train_label
#     return train_data, train_label, dev_data, dev_label, test_data, test_label


def load_data(train_path, dev_path, test_path):
    # train_data, train_label = load_conll_data(train_path)
    # dev_data, dev_label = load_conll_data(dev_path)
    # test_data, test_label = load_conll_data(test_path)
    train_data, train_label = load_my_json_data(train_path)
    dev_data, dev_label = load_my_json_data(dev_path)
    test_data, test_label = load_my_json_data(test_path)
    return train_data, train_label, dev_data, dev_label, test_data, test_label


def data_statistic(path):
    all_data = load_few_shot_data(path, batch_id=-1, word_piece=False, style='all')
    all_support_size = []
    all_support_sent_len = []
    all_test_sent_len = []
    for domain_n, all_batches in all_data:
        for batch in all_batches:
            support_data_set, support_label_set, test_data_set, test_label_set = batch
            all_support_size.append(len(support_data_set))
            for sent in test_data_set:
                all_test_sent_len.append(len(sent))
            for sent in support_data_set:
                all_support_sent_len.append(len(sent))

    def ave(l):
        return sum(l) * 1.0 / len(l)
    print(path)
    print('domains:', [data[0] for data in all_data])
    print('all support data size:', all_support_size)
    print('ave support data size:', ave(all_support_size))
    print('# of support sent', sum(all_support_size))
    print('# of support token ', sum(all_support_sent_len))
    print('# of test token ', sum(all_test_sent_len))
    print('# of all token ', sum(all_support_sent_len) + sum(all_test_sent_len))


def check_few_shot_data(path):
    def label_distribution(label_set):
        all_labels = []
        dup_all_labels = []
        for labels in label_set:
            all_labels.extend(list(set(labels)))
            dup_all_labels.extend(labels)
        return Counter(all_labels), Counter(dup_all_labels)

    for i in range(20):
        support_data_set, support_label_set, test_data_set, test_label_set = load_few_shot_data(path, batch_id=i)
        all_support_label = set([item for sublist in support_label_set for item in sublist])
        all_test_label = set([item for sublist in test_label_set for item in sublist])
        print('batch id', i, 'support num', len(support_data_set), 'test num', len(test_data_set))
        dstrbt, dup_dstrbt = label_distribution(support_label_set)
        dstrbt_t, dup_dstrbt_t = label_distribution(test_label_set)
        print('\t support label', dstrbt)
        print('\t support dup label', dup_dstrbt)
        print('\t test label', dstrbt_t)
        print('\t test dup label', dup_dstrbt_t)
        if not all_support_label == all_test_label:
            print('\tError in data generation:', len(all_support_label), len(all_test_label))
            print('\tLabel only in support', all_support_label-all_test_label)
            print('\tLabel only in test:', all_test_label - all_support_label)


def convert_my_json_data_to_conll_data(input_dir, output_dir, word_piece=True):
    """
    Convert few shot data into 3 conll format data file: source_train, tatget_train, target_test
    """
    os.makedirs(output_dir, exist_ok=True)
    all_file_names = filter(os.path.isfile, os.listdir(input_dir))

    for f_name in all_file_names:
        f_path = os.path.join(input_dir, f_name)

        source_train_seq_ins = []
        source_train_seq_outs = []
        target_train_seq_ins = []
        target_train_seq_outs = []
        target_test_seq_ins = []
        target_test_seq_outs = []

        with open(f_path, 'r') as reader:
            raw_data = json.load(reader)
        for domain_n, domain in raw_data.items():
            # Notice: the batch here means few shot batch, not training batch
            for batch_id, batch in enumerate(domain):
                support_data_set = batch['support']['seq_ins']
                support_label_set = batch['support']['seq_outs']
                support_tokenized_texts = batch['support']['tokenized_texts']
                support_word_piece_labels = batch['support']['word_piece_labels']  # split label for word piece
                support_word_piece_marks = batch['support']['word_piece_marks']  # whether a token is word piece

                test_data_set = batch['batch']['seq_ins']
                test_label_set = batch['batch']['seq_outs']
                test_tokenized_texts = batch['batch']['tokenized_texts']
                test_word_piece_labels = batch['batch']['word_piece_labels']  # split label for word piece
                test_word_piece_marks = batch['batch']['word_piece_marks']  # whether a token is word piece
                if word_piece:
                    if 'train' in f_name:  # construct source_training
                        source_train_seq_ins.extend(support_tokenized_texts)
                        source_train_seq_ins.extend(test_tokenized_texts)

                        source_train_seq_outs.extend(support_word_piece_labels)
                        source_train_seq_outs.extend(test_word_piece_labels)

                        # TODO: extend other data
                        raise NotImplementedError('The coding is not finished')

                    elif 'test' in f_name:  # construct target_training, target_testing
                        # TODO: extend other data
                        pass
                    else:
                        raise RuntimeError('Fail to detect file type')
                else:
                    raise RuntimeError


def make_dev_set_and_move_data_file(args, ori_data_dir, new_data_dir):
    """
    split train to get dev and move all generated data to new place to avoid careless overwrite
    """
    os.makedirs(new_data_dir, exist_ok=True)
    all_file_names = []
    for f in os.listdir(ori_data_dir):
        if os.path.isfile(os.path.join(ori_data_dir, f)):
            all_file_names.append(f)
    print('all files:', os.listdir(ori_data_dir))
    print('Detected files:', all_file_names)
    for file_name in all_file_names:
        input_file_path = os.path.join(ori_data_dir, file_name)
        print('Processing:', input_file_path)
        if 'train' in file_name:
            with open(input_file_path, 'r') as reader:
                ori_train_data = json.load(reader)
            all_data = list(ori_train_data.items())  # 'dict_items' object is not subscriptable
            dev_data = []
            new_train_data = []
            for domain_name, data_item in all_data:
                if domain_name in args.dev_domains:
                    dev_data.append([domain_name, data_item])
                else:
                    new_train_data.append([domain_name, data_item])
            dev_data = dict(dev_data)
            new_train_data = dict(new_train_data)
            new_train_path = os.path.join(new_data_dir, file_name)
            dev_path = os.path.join(new_data_dir, file_name.replace('train', 'dev'))

            with open(new_train_path, 'w') as writer:
                print('Output to:', new_train_path)
                json.dump(new_train_data, writer)
            with open(dev_path, 'w') as writer:
                print('Output to:', new_train_path)
                json.dump(dev_data, writer)

        elif 'test' in file_name:
            print('Copy to:', new_data_dir + file_name)
            shutil.copy(input_file_path, new_data_dir)

        else:
            raise RuntimeError('Unexpected file name')


def read_lines(file_path: str) -> [str]:
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line)
    return lines