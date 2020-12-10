import json
import argparse


def get_filepath(dataset_name: str, dataset_type: str, support_size: int, batch_size: int, batch_num: int):
    return './data/structured_data/Generated.Support_%d.batch_%d.batch_num_%d/%s.%s' \
           '.wp_True.seed_0.support_shots_%d.batch_size_%d.batch_num_%d.json' \
           % (support_size, batch_size, batch_num,
              dataset_name, dataset_type, support_size, batch_size, batch_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display the stats of a dataset')
    parser.add_argument('--dataset_name', type=str,                  required=True)
    parser.add_argument('--dataset_type', type=str, default='train', required=False)
    parser.add_argument('--support_size', type=int, default=1,       required=False)
    parser.add_argument('--batch_size',   type=int, default=20,      required=False)
    parser.add_argument('--batch_num',    type=int, default=100,     required=False)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    dataset_type = args.dataset_type
    support_size = args.support_size
    batch_size = args.batch_size
    batch_num = args.batch_num

    filepath = get_filepath(dataset_name, dataset_type, support_size, batch_size, batch_num)

    data = None
    try:
        f = open(filepath, 'r')
        data = json.load(f)
        f.close()
    except FileNotFoundError:
        print('This dataset cannot be located - %s' % filepath)
        exit()

    domains = data.keys()
    for domain in domains:
        batches = data[domain]
        assert batch_num == len(batches)

        support_set_size = 0
        total_sentence_length = 0

        for batch in batches:
            assert batch_size == len(batch['batch']['seq_ins'])

            support_set_size += len(batch['support']['seq_ins'])

            for seq_in in batch['batch']['seq_ins']:
                total_sentence_length += len(seq_in)
            for seq_in in batch['support']['seq_ins']:
                total_sentence_length += len(seq_in)

        avg_support_set_size = support_set_size / batch_num
        avg_sentence_length = total_sentence_length / (batch_num * batch_size + support_set_size)

        print('Dataset Info on %s - %s\t%d\t%d\t%d' % (domain.upper(), dataset_type, support_size, batch_size, batch_num))
        print('avg_support_set_size\t\t%.2f' % avg_support_set_size)
        print('avg_sentence_length\t\t%.2f' % avg_sentence_length)
        print()
