# coding: utf-8
import json
import argparse
import sys
import os
import random
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from utils.data_loader import *
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer
from config import *


def conll_format_output():
    pass
    raise NotImplementedError


def json_format_output_similarity(opt, dataset_name, dataset, mark='few_shot'):
    """
    input1: 'dataset_name'
    input2: {
                'data_part_name1':[{'seq_ins':[], 'labels'[]:, 'seq_outs':[]}, ..., batch_n]
                'data_part_name2':[{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}, ..., batch_n]
                'data_part_name3':[{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}, ..., batch_n]
            }
    """
    output_dir = opt.output_dir
    print('Target dir: ', output_dir)
    # for dataset_name, dataset in dataset.items():
    for data_part_name, data_part in dataset.items():
        l_shot_mark = '.support_shots_{}'.format(opt.support_shots) if data_part_name == 'train' else ''
        t_shot_mark = '.batch_size_{}'.format(opt.batch_size) if data_part_name == 'test' else ''
        file_name = "{}.{}.{}.size_{}.seed_{}{}{}.json".format(
            dataset_name, mark, data_part_name, len(data_part['labels']), opt.seed, l_shot_mark, t_shot_mark
        )
        dir_name = 'Support_{}_shot.batch_{}_shot'
        sub_dir_path = os.path.join(output_dir, dir_name)

        file_path = os.path.join(sub_dir_path, file_name)
        print('Output to: {}'.format(file_name))
        with open(file_path, 'w') as writer:
            json.dump(data_part, writer)


def load_data(opt):
    """
    Load all dataset into a dict var, example:
    {
        'dataset_name1':{
            'train':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
            'test':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
            'valid':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
        }
        'dataset_name2':{
            'train':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
            'test':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
            'valid':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
        }
    }
    """

    all_data = {}
    for name in opt.name_lst:
        one_dataset = {}
        print('Start loading data of: ', name )
        for part_name in ['train', 'valid', 'test']:
            one_part = {}
            seq_in_path = '{}{}/{}/{}'.format(opt.raw_dir, name, part_name, 'seq.in')
            label_path = '{}{}/{}/{}'.format(opt.raw_dir, name, part_name, 'label')
            seq_out_path = '{}{}/{}/{}'.format(opt.raw_dir, name, part_name, 'seq.out')
            seq_ins, labels, seq_outs = [], [], []
            # with open(seq_in_path, 'r', encoding='utf-8') as reader:
            #     debuger = reader.readlines()
            #     if len(debuger) != len(set(debuger)):
            #         print('Warning!!', len(debuger), len(set(debuger)))
            #         ept = set()
            #         for s in debuger:
            #             if s in ept:
            #                 print('duplicate: ', s)
            #             else:
            #                 ept.add(s)
            #         raise RuntimeError('There is duplicate in dataset')
            with open(seq_in_path, 'r', encoding='utf-8') as reader:
                for line in reader:
                    seq_ins.append(line.replace('\n', '').split())
            with open(seq_out_path, 'r', encoding='utf-8') as reader:
                for line in reader:
                    seq_outs.append(line.replace('\n', '').split())
            with open(label_path, 'r', encoding='utf-8') as reader:
                for line in reader:
                    labels.append(line.replace('\n', '').split('#'))
            one_part['seq_ins'] = seq_ins
            one_part['seq_outs'] = seq_outs
            one_part['labels'] = labels
            one_dataset[part_name] = one_part
            print(part_name, [(x, len(y)) for x, y in one_part.items()])
        all_data[name] = one_dataset
    return all_data


def json_format_output_few_shot(opt, dataset_name, dataset, mark='few_shot'):
    """
    input1: 'dataset_name'
    input2: { # one dataset
                'data_part_name1':[
                { # one_batch
                    'support':{'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
                    'batch':{'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
                },
                ...,batch_n],
                'data_part_name2':[batch1, ..., batch_n]
            }
    """
    output_dir = opt.output_dir
    print('Target dir: ', output_dir)
    file_name = "{}.{}.seed_{}.support_shots_{}.batch_size_{}.batch_num_{}.json".format(
        dataset_name, mark, opt.seed, opt.support_shots, opt.batch_size, opt.batch_num
    )
    dir_name = 'Generated.Support_{}.batch_{}.batch_num_{}'.format(opt.support_shots, opt.batch_size, opt.batch_num)
    sub_dir_path = os.path.join(output_dir, dir_name)
    os.makedirs(sub_dir_path, exist_ok=True)
    file_path = os.path.join(sub_dir_path, file_name)
    print('Output to: {}'.format(file_name))
    # if os.path.exists(file_path):
    #     raise ValueError("Output file already exists and is not empty.")
    with open(file_path, 'w') as writer:
        json.dump(dataset, writer)


def gen_domain_trans_style_data(opt):
    print('Setting is abandoned')
    raise NotImplementedError


def all_slots_from_data_part(seq_outs):
    ret = set([])
    for seq_out in seq_outs:
        ret = ret | set(seq_out)
    return ret


def sample_one_data_part(opt, data_part, result_shot_num, ignoring_slot_names):
    """
    transfer full-data into few-shot-data by sampling for one data part
    input:  { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
    input:  data shots for each label
    output: result few shot data part
            { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
    output: remained data part
            { 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
    """
    print('START sample_one_data_part')
    data_part_slots = all_slots_from_data_part(data_part['seq_outs'])
    ''' Extract data id into slot buckets'''
    split_data = {sn: [] for sn in data_part_slots}  # data bucket: {slot_name:[[data_id, data_item]]}
    idx2sequence_slot_names = {}
    for ind, seq_out in enumerate(data_part['seq_outs']):
        slot_set = set(seq_out)
        if opt.min_slot_appear > 1 and slot_set & set(ignoring_slot_names):
            continue  # abandon samples with bad slot type
        for slot in slot_set:  # iter within unique slot labels for each sample
            if split_data[slot]:
                split_data[slot].append([ind, slot_set])
            else:
                split_data[slot] = [[ind, slot_set]]
            idx2sequence_slot_names[ind] = slot_set

    ''' Sample learning shots, and record the selected data's id '''
    shot_nums = {sn: 0 for sn in split_data}
    selected_item_ids = set()  # record appeared data item id

    def not_appeared(item):
        if not (0 <= item[0] < len(data_part['seq_outs'])):
            raise RuntimeError
        return False if item[0] in selected_item_ids else True

    def update_slot_nums(item):
        for slot in item[1]:
            shot_nums[slot] += 1

    # Old version that does not conform to the algorithm described in the paper
    # for i in range(result_shot_num):  # collect the i_th shot
    #     for slot, items in split_data.items():  # for each shot type
    #         fresh_items = list(filter(not_appeared, items))
    #         if i + 1 < shot_nums[slot]:  # the i_th shot for this slot has been collected
    #             pass
    #         elif len(list(fresh_items)) == 0:  # there are no sufficient shots
    #             pass
    #         else:
    #             item = random.choice(fresh_items)
    #             selected_item_ids.add(item[0])
    #             update_slot_nums(item)

    print('Divide data')
    kept_slot_names = data_part_slots - set(ignoring_slot_names)

    # Version 1 for Part 1 of the minimum-including algorithm.
    for slot_name in kept_slot_names:
        idxes = list(idx2sequence_slot_names.keys())
        num_trials = 0
        while shot_nums[slot_name] < result_shot_num:
            num_trials += 1
            if num_trials == 50:
                break
            random.shuffle(idxes)

            for idx in idxes:
                slot_names = idx2sequence_slot_names[idx]
                if not (idx in selected_item_ids or len(slot_names) == 0 or slot_name not in slot_names):
                    selected_item_ids.add(idx)
                    update_slot_nums((idx, slot_names))
                    break

    # Version 2 for Part 1 of the minimum-including algorithm.
    # while True:
    #     should_continue_iterating_while = False
    #     for slot_name in kept_slot_names:
    #         if slot_name not in shot_nums.keys() or shot_nums[slot_name] < result_shot_num:
    #             should_continue_iterating_while = True
    #
    #     if not should_continue_iterating_while:
    #         break
    #
    #     for slot_name in kept_slot_names:
    #         should_continue_iterating_for = False
    #         for tmp_slot_name in kept_slot_names:
    #             if tmp_slot_name not in shot_nums.keys() or shot_nums[tmp_slot_name] < result_shot_num:
    #                 should_continue_iterating_for = True
    #
    #         if not should_continue_iterating_for:
    #             break
    #
    #         idxes = list(idx2sequence_slot_names.keys())
    #         random.shuffle(idxes)
    #
    #         for idx in idxes:
    #             slot_names = idx2sequence_slot_names[idx]
    #             if not (idx in selected_item_ids or len(slot_names) == 0 or slot_name not in slot_names):
    #                 selected_item_ids.add(idx)
    #                 update_slot_nums((idx, slot_names))
    #                 break

    print('result_shot_num'.upper(), result_shot_num)
    print('THE NUMBER OF selected_item_ids BEFORE REMOVAL', len(selected_item_ids))

    # Part 2 of the minimum-including algorithm with non-deterministic removal of extra samples
    selected_item_ids = list(selected_item_ids)
    for idx in selected_item_ids:
        to_be_removed_slot_names = idx2sequence_slot_names[idx]
        can_remove = True
        for to_be_removed_slot_name in to_be_removed_slot_names:
            if shot_nums[to_be_removed_slot_name] - 1 < result_shot_num:
                can_remove = False
                break
        if can_remove:
            if random.randint(1, 100) < 80:
                selected_item_ids.remove(idx)
                for to_be_removed_slot_name in to_be_removed_slot_names:
                    shot_nums[to_be_removed_slot_name] -= 1

    selected_item_ids = set(selected_item_ids)
    print('THE NUMBER OF selected_item_ids AFTER REMOVAL', len(selected_item_ids))
    if len(selected_item_ids) < 3:
        for id in selected_item_ids:
            print(idx2sequence_slot_names[id])


    ''' Pick data item by selected id '''
    few_shot_data = {'seq_ins': [], 'labels': [], 'seq_outs': []}
    for item_id in selected_item_ids:
        few_shot_data['seq_ins'].append(data_part['seq_ins'][item_id])
        few_shot_data['seq_outs'].append(data_part['seq_outs'][item_id])
        few_shot_data['labels'].append(data_part['labels'][item_id])

    ''' Collect remained data by un-selected id '''
    remained_data = {'seq_ins': [], 'labels': [], 'seq_outs': []}
    for item_id in range(len(data_part['labels'])):
        if item_id not in selected_item_ids:
            remained_data['seq_ins'].append(data_part['seq_ins'][item_id])
            remained_data['seq_outs'].append(data_part['seq_outs'][item_id])
            remained_data['labels'].append(data_part['labels'][item_id])
    if (len(few_shot_data['labels']) + len(remained_data['labels']) != len(data_part['labels'])) or len(few_shot_data) == 0 or len(remained_data) == 0:
        raise RuntimeError("ERROR: Failed to collect remained or few shot data: few_shot {} remained {} total{}".format(
            len(few_shot_data['labels']), len(remained_data['labels']), len(data_part['labels'])))
    print('Returning data')
    return few_shot_data, remained_data


def get_test_batch(data_part, batch_size):
    idxes = [i for i in range(len(data_part['seq_ins']))]
    random.shuffle(idxes)
    test_batch = {'seq_ins': [], 'labels': [], 'seq_outs': []}

    for i in range(batch_size):
        idx = idxes[i]
        test_batch['seq_ins'].append(data_part['seq_ins'][idx])
        test_batch['labels'].append(data_part['labels'][idx])
        test_batch['seq_outs'].append(data_part['seq_outs'][idx])

    return test_batch


def get_ignoring_slot_names(opt, data_part):
    """ get a list of slot names, that should be ignore in fewshot data generation """
    ignoring_slot_names = []
    data_part_slots = all_slots_from_data_part(data_part['seq_outs'])
    ''' Extract data id into slot buckets'''
    split_data = {sn: [] for sn in data_part_slots}  # data bucket: {slot_name:[[data_id, data_item]]}
    for ind, seq_out in enumerate(data_part['seq_outs']):
        slot_set = set(seq_out)
        for slot in slot_set:  # iter within unique slot labels for each sample
            if split_data[slot]:
                split_data[slot].append([ind, slot_set])
            else:
                split_data[slot] = [[ind, slot_set]]
    ''' filter out slots that has too few samples (Avoid useless support) '''
    if opt.min_slot_appear > 1:
        # print('!!!!!!! debug, before filtering bucket',
        #       sorted([[sn, len(split_data[sn])] for sn in split_data], key=lambda x: x[1]))
        filtered_split_data = {}
        for slot_name, bucket in split_data.items():
            if len(bucket) >= opt.min_slot_appear:
                filtered_split_data[slot_name] = bucket
            else:
                ignoring_slot_names.append(slot_name)
        split_data = filtered_split_data
        # print('####### debug, after filtering bucket',
        #       sorted([[sn, len(split_data[sn])] for sn in split_data], key=lambda x: x[1]))
    return ignoring_slot_names


def gen_few_shot_data(opt, split_data):
    """
    Generate the few shot data
    output: few shot data for each domain
    {
        'dataset_name1':{
            'intent/domain_name1':[
                    # one batch
                    {
                        'support':{'seq_ins':[], 'labels':[], 'seq_outs':[]},
                        'batch':{'seq_ins':[], 'labels':[], 'seq_outs':[]},
                    },
                    ...,
                    batch_n
                ]
            }
            'intent/domain_name2':[batch1, batch2, ..., batch_n]
        }
        'dataset_name2':{ ... }
    }
    """
    all_few_shot_data = {}
    for dataset_name, dataset in split_data.items():
        few_shot_data = {}

        ''' Sample one domain's few shot data '''
        for domain_name, domain in dataset.items():
            ''' do filtering '''
            if len(domain['labels']) < opt.min_data:  # out bad domains (for atis)
                continue
            ignoring_slot_names = get_ignoring_slot_names(opt, domain)
            batches = []
            ''' sample batches '''
            for batch_id in range(opt.batch_num):
                ''' sample support set '''
                # print(''' debug: get support set ''')
                support_set, remained_data = sample_one_data_part(opt, domain, opt.support_shots, ignoring_slot_names)
                ''' randomly sample batch data, batch data are not appeared in support set '''
                # print(''' debug: get test set ''')
                # test_batch, _ = sample_one_data_part(opt, remained_data, opt.batch_size, ignoring_slot_names)
                test_batch = get_test_batch(remained_data, opt.batch_size)
                # ''' intersection check '''
                # for seq_in in few_shot_test['seq_ins']:
                #     if seq_in in few_shot_support['seq_ins']:
                #         raise RuntimeError('Warning: There is a intersection between support and test set')
                batches.append({'support': support_set, 'batch': test_batch})
                if batch_id % 20 == 0:
                    print('\tfor domain:', domain_name, batch_id, 'batches finished')
            one_domain_few_shot_data = batches
            few_shot_data[domain_name] = one_domain_few_shot_data
        all_few_shot_data[dataset_name] = few_shot_data
    return all_few_shot_data


def gen_similarity_method_data(opt, all_data):
    """ sample training data: each slot at least has x shot data, example return var:
        output: {
            'dataset_name1':{
                'train':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
                'test':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
                'valid':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
            }
            'dataset_name2':{
                'train':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
                'test':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
                'valid':{ 'seq_ins':[], 'labels'[]:, 'seq_outs':[]}
            }
        }
    """
    all_few_shot_data = {}
    for data_name, dataset in all_data.items():
        print('\nSampling few shot data from dataset:', data_name)
        few_shot_data = {'train': {}, 'test': dataset['test'], 'valid': dataset['valid']}
        train_data = dataset['train']
        test_data = dataset['test']
        valid_data = dataset['valid']

        train_slots = all_slots_from_data_part(train_data['seq_outs'])
        test_slots = all_slots_from_data_part(test_data['seq_outs'])
        valid_slots = all_slots_from_data_part(valid_data['seq_outs'])

        if valid_slots - train_slots or test_slots - train_slots:
            print('Warning: there is unseen slots in valid or test')
            print('Valid - Train:', valid_slots - train_slots)
            print('Test - Train:', test_slots - train_slots)
            # raise Warning('There is unseen slots in valid or test')
        else:
            print('There is no unseen slots in valid and test')

        # obtain few shot data
        few_shot_train_data, _ = sample_one_data_part(opt, train_data, opt.support_shots, [])
        few_shot_data['train'] = few_shot_train_data

        all_few_shot_data[data_name] = few_shot_data

        ''' Collect log info '''
        all_slots = train_slots | test_slots | valid_slots
        print('Total slot label num: {}, train slots num: {}'.format(len(all_slots), len(train_slots)))
        for part_name in few_shot_data:
            print(part_name, [(x, len(y)) for x, y in few_shot_data[part_name].items()])

    return all_few_shot_data


def split_domains(opt, all_data):
    """ Split data according to intents & domains, example
    output:
    {
        'dataset_name1':{
            'intent/domain_name1':{
                'seq_ins':[], 'labels'[]:, 'seq_outs':[]
            }
            'intent/domain_name2':{
                'seq_ins':[], 'labels'[]:, 'seq_outs':[]
            }
        }
        'dataset_name2':{
            ...
        }
    }
    """
    def split_one_data_part(data_part):
        all_domains = {}
        if not(len(data_part['seq_ins']) == len(data_part['seq_outs']) == len(data_part['labels'])):
            print('ERROR: seq_ins, seq_outs, labels are not equal in amount')
            raise RuntimeError
        for item in zip(data_part['seq_ins'], data_part['seq_outs'], data_part['labels']):
            seq_in, seq_out, labels = item[0], item[1], item[2]
            for label in labels:  # multi intent is split by '#'
                if label in all_domains:
                    all_domains[label]['seq_ins'].append(seq_in)
                    all_domains[label]['seq_outs'].append(seq_out)
                    all_domains[label]['labels'].append(label)
                else:
                    all_domains[label] = {}
                    all_domains[label]['seq_ins'] = [seq_in]
                    all_domains[label]['seq_outs'] = [seq_out]
                    all_domains[label]['labels'] = [label]
        return all_domains

    def merge_data_part(train_data, test_data, valid_data):
        merged_data = {}
        all_data_parts = [train_data, valid_data, test_data]
        for data_part in all_data_parts:
            for domain_name in data_part:
                if domain_name in merged_data:
                    merged_data[domain_name]['seq_ins'].extend(data_part[domain_name]['seq_ins'])
                    merged_data[domain_name]['seq_outs'].extend(data_part[domain_name]['seq_outs'])
                    merged_data[domain_name]['labels'].extend(data_part[domain_name]['labels'])
                else:
                    merged_data[domain_name] = data_part[domain_name]
        return merged_data

    split_data = {}
    for dataset_name in all_data:
        print('Start split', dataset_name)
        splited_train_data = split_one_data_part(all_data[dataset_name]['train'])
        splited_test_data = split_one_data_part(all_data[dataset_name]['test'])
        splited_valid_data = split_one_data_part(all_data[dataset_name]['valid'])

        # split_data[dataset_name] = reconstruct_data(splited_train_data, splited_test_data, splited_valid_data,
        #                                               dataset_name)

        split_data[dataset_name] = merge_data_part(splited_train_data, splited_test_data, splited_valid_data)
    return split_data


def data_statistics(split_data):
    """ calculate how many domains, labels, how many data and label in each domain"""
    print('=== Start statistic ====')
    for dataset_name, dataset in split_data.items():
        domains_num = len(dataset)
        total_label_set = set()
        statistic_items = []
        for d_n, d in dataset.items():
            flatten_labels = [item for sublist in d['seq_outs'] for item in sublist]
            label_set = set([label.replace('B-', '').replace('I-', '') for label in set(flatten_labels)])
            total_label_set = total_label_set | label_set
            statistic_item = (d_n, len(d['labels']), len(label_set), label_set)
            statistic_items.append(statistic_item)
        for item in sorted(statistic_items, key=lambda x: x[1]):
            print('Domain: {},\tdata num: {},\tlabel num: {},\tlabels:{}'.format(*item))
        print('Total domains:', domains_num)
        print('Total labels:', len(total_label_set))


def reform_data_into_word_piece_style(opt, few_shot_data):
    """ tokenize and pre-process the dataset into word piece style: split label according to wordpiece """
    tokenizer = BertTokenizer.from_pretrained(opt.bert_vocab)
    for dataset_name, dataset in few_shot_data.items():
        print('Start to word piecing dataset: {}'.format(dataset_name))
        for domain_name, domain in dataset.items():
            print('\tprocessing domain: {}'.format(domain_name))
            for batch in domain:
                for data_part_name, data_part in batch.items():
                    word_piece_marks, segment_ids, tokenized_texts, word_piece_labels = \
                        get_word_piece_labeled_data(tokenizer, data_part['seq_ins'], data_part['seq_outs'])
                    # data_part['indexed_datas'] = indexed_datas
                    data_part['word_piece_marks'] = word_piece_marks
                    data_part['tokenized_texts'] = tokenized_texts
                    data_part['word_piece_labels'] = word_piece_labels


def main():
    parser = argparse.ArgumentParser()
    DEFAULT_RAW_DIR = './data/raw_data/'
    DEFAULT_DATA_DIR = './data/structured_data/'
    BERT_BASE_UNCASED_VOCAB = './data/bert-base-uncased-vocab.txt'
    # file path
    parser.add_argument("--raw_dir", type=str, default=DEFAULT_RAW_DIR, help="path to the raw data dir")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_DATA_DIR, help="path to the result data dir")
    parser.add_argument("--name_lst", nargs='+', default=['atis'], help='dataset names to be processed')
    parser.add_argument("--bert_vocab", type=str, default=BERT_BASE_UNCASED_VOCAB,
                     help="pass a path to vocab file")

    # data size
    parser.add_argument('--support_shots', type=int, default=1,
                        help='learning shots for few shot learning style')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='test shots for few shot learning style')
    parser.add_argument('--batch_num', type=int, default=100,
                        help='train batches for few shot learning style')
    parser.add_argument('--min_data', type=int, default=100,
                        help='(for atis) filter out domains that have data amount less than min data')
    parser.add_argument('--min_slot_appear', type=int, default=2,
                        help='(for atis) filter out slot type that appears less this value, '
                             'other wise will get useless support')

    # generate content for speeding up
    parser.add_argument('--target_only', action='store_true', help='only generate target train sets')

    # general setting
    parser.add_argument('-nwp', '--no_word_piece', action='store_true', help='weather generate word piece label & data')
    parser.add_argument("--style_lst", nargs='+', default=['few_shot'], help="output data styles, choice: similarity, few_shot")
    parser.add_argument("--test_domains", nargs='+', default=['GetWeather', 'atis_ground_service'], help="choose which domain is used to generate test data")
    parser.add_argument('-sd', '--seed', type=int, default=0, help='random seed, do nothing when sd < 0')
    opt = parser.parse_args()
    print('Parameter:\n', json.dumps(vars(opt), indent=2))

    if opt.seed >= 0:
        random.seed(opt.seed)

    all_data = load_data(opt)

    if 'similarity' in opt.style_lst:
        similarity_method_data = gen_similarity_method_data(opt, all_data)
        # with open('./debug_few_shot_data.json', 'w') as writer:
        #     json.dump(similarity_method_data, writer)
        for dataset_name, dataset in similarity_method_data:
            json_format_output_similarity(opt, dataset_name, dataset, mark='similarity')

    # if 'domain_trans' in opt.style_lst:
    #     gen_domain_trans_style_data(opt, all_data)

    split_data = split_domains(opt, all_data)

    if 'few_shot' in opt.style_lst:
        print('start generate few shot data.')
        print('START TO GATHER FEW_SHOT_DATA')
        few_shot_data = gen_few_shot_data(opt, split_data)
        print('FEW_SHOT_DATA GATHERED')
        if not opt.no_word_piece:
            reform_data_into_word_piece_style(opt, few_shot_data)

        print('start to dump data')
        for dataset_name, dataset in few_shot_data.items():
            train_set, test_set = {}, {}
            ''' split test and train domain '''
            for domain_name, domain in dataset.items():
                if domain_name in opt.test_domains:
                    test_set[domain_name] = domain
                else:
                    train_set[domain_name] = domain
            json_format_output_few_shot(opt, dataset_name, train_set, mark='train.wp_{}'.format(not opt.no_word_piece))
            json_format_output_few_shot(opt, dataset_name, test_set, mark='test.wp_{}'.format(not opt.no_word_piece))

    # json_format_output(opt, split_data)
    data_statistics(split_data)


if __name__ == "__main__":
    main()

