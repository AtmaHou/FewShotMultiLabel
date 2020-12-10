#!/bin/bash

if [ -f dataset_info.txt ]; then
	rm dataset_info.txt
fi

touch dataset_info.txt

python check_dataset_info.py --dataset_name conll2003 --dataset_type train --support_size 5 --batch_size 10 --batch_num 100 >> dataset_info.txt

python check_dataset_info.py --dataset_name ontonotes --dataset_type train --support_size 5 --batch_size 10 >> dataset_info.txt

python check_dataset_info.py --dataset_name gum --dataset_type train --support_size 5 --batch_size 10 --batch_num 100 >> dataset_info.txt

python check_dataset_info.py --dataset_name wnut17 --dataset_type train --support_size 5 --batch_size 10 --batch_num 100 >> dataset_info.txt

python check_dataset_info.py --dataset_name snips --dataset_type train --support_size 5 --batch_size 10 >> dataset_info.txt

python check_dataset_info.py --dataset_name snips --dataset_type test --support_size 5 --batch_size 10 >> dataset_info.txt
