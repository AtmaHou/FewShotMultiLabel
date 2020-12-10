import argparse


dir_path = './data/raw_data/'


def find_spans(tags: [str]) -> [[str]]:
    spans = []
    span = []
    prev_prefix = ''
    for tag in tags:
        if tag == 'O':
            if span:
                spans.append(span)
                span = []
                prev_prefix = 'O'
        if tag.startswith('B'):
            if prev_prefix == 'I':
                spans.append(span)
                span = []
            elif prev_prefix == 'B':
                spans.append(span)
                span = []
            span.append(tag)
            prev_prefix = 'B'
        if tag.startswith('I'):
            span.append(tag)

    if span:
        spans.append(span)

    # print('tags ', tags)
    # print('spans', spans)
    # print()
    return spans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the average length of spans of domains in a dataset')
    parser.add_argument('--datasets', type=str)

    args = parser.parse_args()

    datasets = [args.datasets]

    for dataset in datasets:
        train_path = dir_path + dataset + '/train/label'
        valid_path = dir_path + dataset + '/valid/label'
        test_path = dir_path + dataset + '/test/label'

        train_domains = []
        valid_domains = []
        test_domains = []

        with open(train_path) as f:
            for line in f:
                train_domains.append(line.strip('\n'))

        with open(valid_path) as f:
            for line in f:
                valid_domains.append(line.strip('\n'))

        with open(test_path) as f:
            for line in f:
                test_domains.append(line.strip('\n'))

        total_domains = train_domains + valid_domains + test_domains

        train_path = dir_path + dataset + '/train/seq.out'
        valid_path = dir_path + dataset + '/valid/seq.out'
        test_path = dir_path + dataset + '/test/seq.out'

        train_lines = []
        valid_lines = []
        test_lines = []

        with open(train_path) as f:
            for line in f:
                train_lines.append(line)

        with open(valid_path) as f:
            for line in f:
                valid_lines.append(line)

        with open(test_path) as f:
            for line in f:
                test_lines.append(line)

        total_lines = train_lines + valid_lines + test_lines

        assert len(total_domains) == len(total_lines)

        domain2spans = dict()

        for i in range(len(total_domains)):
            domain = total_domains[i]
            line = total_lines[i]
            if domain not in domain2spans:
                domain2spans[domain] = []
            domain2spans[domain].append(find_spans(line.split()))

        print('The information on average tag span length for %s is'
              % dataset)
        for domain in domain2spans.keys():
            if len(domain) >= 16:
                print('%s\t%.2f' % (domain, sum([len(span) for span in domain2spans[domain]]) / len(domain2spans[domain])))
            else:
                print('%s\t\t%.2f' % (domain, sum([len(span) for span in domain2spans[domain]]) / len(domain2spans[domain])))


