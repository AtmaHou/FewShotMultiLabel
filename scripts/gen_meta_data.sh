#!/usr/bin/env bash

echo usage: pass dataset list as param, split with space
echo eg: source gen_mate_data.sh atis

dataset_lst=(toursg)
#dataset_lst=(stanford)
#dataset_lst=(atis stanford toursg)

# ======= size setting ======
support_shots_lst=(1 5)
#support_shots_lst=(5)
#episode_num=1000  # We could over generation and select part of for each epoch
episode_num=500
query_shot=16
word_piece_data=True
way=-1

remove_rate=80


# ====== general setting =====
seed_lst=(0)

task=mlc

#dup_query=--dup_query  # dup_query set empty to not allow duplication between query and support
dup_query=

allow_override=--allow_override

check=--check

# ====== train & test setting ======
split_basis=domain

#eval_confif_id_lst=(1)  # for snips
eval_config_id_lst=(0 1 2 3 4 5)  # for toursg
label_type_lst=(attribute)

# ======= default path (for quick distribution) ==========
input_dir=/the/path/of/dstc4/data/
output_dir=./MetaData_Debug/
eval_config_dir=./other_tool/config/

echo \[START\] set jobs on dataset \[ ${dataset_lst[@]} \]
# === Loop for all case and run ===
for seed in ${seed_lst[@]}
do
  for dataset in ${dataset_lst[@]}
  do
    for support_shots in ${support_shots_lst[@]}
    do
        for eval_config_id in ${eval_config_id_lst[@]}
        do
            for label_type in ${label_type_lst[@]}
            do
              echo start generate with the config \[ ${eval_config_id} \] with label type \[ ${label_type} \]
              input_path=${input_dir}
              mark=try
              eval_config=${eval_config_dir}/config${eval_config_id}.json

              export OMP_NUM_THREADS=2  # threads num for each task
              python3 ./other_tool/meta_dataset_generator/generate_meta_dataset.py \
                --input_path ${input_path} \
                --output_dir ${output_dir} \
                --dataset ${dataset} \
                --episode_num ${episode_num} \
                --support_shots ${support_shots} \
                --query_shot ${query_shot} \
                --way ${way} \
                --task ${task} \
                --seed ${seed} \
                --split_basis ${split_basis} \
                --eval_config ${eval_config} \
                --eval_config_id ${eval_config_id} \
                --label_type ${label_type} \
                --remove_rate ${remove_rate} \
                --mark ${mark} ${dup_query} ${allow_override} ${check} > ${output_dir}${dataset}.${eval_config_id}.spt_s_${support_shots}.q_s_${query_shot}.ep_${episode_num}.log
            done
        done
    done
  done
done





