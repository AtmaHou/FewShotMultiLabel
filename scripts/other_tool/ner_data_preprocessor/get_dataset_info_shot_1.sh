#!/bin/bash

if [ -f dataset_info.txt ]; then
	rm dataset_info.txt
fi

touch dataset_info.txt

python check_dataset_info.py --dataset_name conll2003 --dataset_type train --support_size 1 --batch_size 20 --batch_num 200 >> dataset_info.txt
python check_dataset_info.py --dataset_name conll2003 --dataset_type train --support_size 2 --batch_size 20 --batch_num 200 >> dataset_info.txt
python check_dataset_info.py --dataset_name conll2003 --dataset_type train --support_size 3 --batch_size 20 --batch_num 200 >> dataset_info.txt
python check_dataset_info.py --dataset_name conll2003 --dataset_type train --support_size 5 --batch_size 20 --batch_num 200 >> dataset_info.txt
python check_dataset_info.py --dataset_name conll2003 --dataset_type train --support_size 10 --batch_size 20 --batch_num 200 >> dataset_info.txt

python check_dataset_info.py --dataset_name anem --dataset_type train --support_size 1 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name anem --dataset_type train --support_size 2 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name anem --dataset_type train --support_size 3 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name anem --dataset_type train --support_size 5 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name anem --dataset_type train --support_size 10 --batch_size 20 >> dataset_info.txt

python check_dataset_info.py --dataset_name sec-fillings --dataset_type train --support_size 1 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name sec-fillings --dataset_type train --support_size 2 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name sec-fillings --dataset_type train --support_size 3 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name sec-fillings --dataset_type train --support_size 5 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name sec-fillings --dataset_type train --support_size 10 --batch_size 20 >> dataset_info.txt

python check_dataset_info.py --dataset_name ontonotes --dataset_type train --support_size 1 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name ontonotes --dataset_type train --support_size 2 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name ontonotes --dataset_type train --support_size 3 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name ontonotes --dataset_type train --support_size 5 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name ontonotes --dataset_type train --support_size 10 --batch_size 20 >> dataset_info.txt

python check_dataset_info.py --dataset_name re3d --dataset_type train --support_size 1 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name re3d --dataset_type train --support_size 2 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name re3d --dataset_type train --support_size 3 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name re3d --dataset_type train --support_size 5 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name re3d --dataset_type train --support_size 10 --batch_size 20 >> dataset_info.txt

python check_dataset_info.py --dataset_name wikigold --dataset_type train --support_size 1 --batch_size 20 --batch_num 200 >> dataset_info.txt

python check_dataset_info.py --dataset_name gum --dataset_type train --support_size 1 --batch_size 20 --batch_num 200 >> dataset_info.txt

python check_dataset_info.py --dataset_name wnut17 --dataset_type train --support_size 1 --batch_size 20 --batch_num 200 >> dataset_info.txt
python check_dataset_info.py --dataset_name wnut17 --dataset_type train --support_size 2 --batch_size 20 --batch_num 200 >> dataset_info.txt
python check_dataset_info.py --dataset_name wnut17 --dataset_type train --support_size 3 --batch_size 20 --batch_num 200 >> dataset_info.txt
python check_dataset_info.py --dataset_name wnut17 --dataset_type train --support_size 5 --batch_size 20 --batch_num 200 >> dataset_info.txt
python check_dataset_info.py --dataset_name wnut17 --dataset_type train --support_size 10 --batch_size 20 --batch_num 200 >> dataset_info.txt

python check_dataset_info.py --dataset_name snips --dataset_type train --support_size 1 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name snips --dataset_type train --support_size 2 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name snips --dataset_type train --support_size 3 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name snips --dataset_type train --support_size 5 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name snips --dataset_type train --support_size 10 --batch_size 20 >> dataset_info.txt

python check_dataset_info.py --dataset_name snips --dataset_type test --support_size 1 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name snips --dataset_type test --support_size 2 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name snips --dataset_type test --support_size 3 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name snips --dataset_type test --support_size 5 --batch_size 20 >> dataset_info.txt
python check_dataset_info.py --dataset_name snips --dataset_type test --support_size 10 --batch_size 20 >> dataset_info.txt
