#!/usr/bin/env python3
# -*- coding: utf8 -*-
#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
get the parser tag data from origin data to statistic the grammar information for our model
"""

import os
import json
from nltk.tag import StanfordPOSTagger


MODEL_DIR='/the_path_of_your_stanford_postagger_model/'
DATA_DIR = '/the_path_of_your_few_shot_training_data_path/'

DEBUG = False


def load_tagger(model_dir):
    tagger_model_file = os.path.join(model_dir, 'models/english-bidirectional-distsim.tagger')
    tagger_jar_file = os.path.join(model_dir, 'stanford-postagger.jar')
    tagger = StanfordPOSTagger(model_filename=tagger_model_file, path_to_jar=tagger_jar_file)
    return tagger


def get_tag_data(tagger, prefix=''):
    all_sub_dirs = os.listdir(DATA_DIR)
    all_sub_dirs = [sub_dir for sub_dir in all_sub_dirs if sub_dir.startswith(prefix)]
    print('all_sub_dirs: {}'.format(all_sub_dirs))
    json_file_lst = ['train.json', 'dev.json', 'test.json']

    res = {}
    d_count = 0

    if DEBUG:
        all_sub_dirs = all_sub_dirs[:1]
        json_file_lst = ['dev.json']

    for sub_dir in all_sub_dirs:
        for json_file in json_file_lst:
            filename = os.path.join(DATA_DIR, sub_dir, json_file)
            with open(filename, 'r') as fr:
                json_data = json.load(fr)
            for key, episode_data in json_data.items():
                for e_item in episode_data:
                    for seq_in in e_item['support']['seq_ins']:
                        text = ' '.join(seq_in)
                        if text not in res:
                            d_count += 1
                            tag_data = tagger.tag(seq_in)
                            res[text] = tag_data
                            if d_count % 10 == 0:
                                print('d_count - {}'.format(d_count))

                    for seq_in in e_item['query']['seq_ins']:
                        text = ' '.join(seq_in)
                        if text not in res:
                            d_count += 1
                            tag_data = tagger.tag(seq_in)
                            res[text] = tag_data
                            if d_count % 10 == 0:
                                print('d_count - {}'.format(d_count))
    return res


def store_data(data, store_file):
    with open(store_file, 'w') as fw:
        json.dump(data, fw)
    print('store successfully!')


if __name__ == '__main__':
    store_file = os.path.join(DATA_DIR, 'tag_data.dict' + ('.all' if not DEBUG else '.debug'))
    tagger = load_tagger(MODEL_DIR)
    res = get_tag_data(tagger)
    store_data(res, store_file)


