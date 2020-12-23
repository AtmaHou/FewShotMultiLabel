# Few-shot MLC

The code of AAAI2021 paper [Few-Shot Learning for Multi-label Intent Detection](https://arxiv.org/abs/2010.05256).

The code framework is based on few-shot learning platform: [MetaDialog](https://github.com/AtmaHou/MetaDialog).

## Get Started
### Requirement
```
python >= 3.6
pytorch >= 1.5.0
transformers >= 2.8.0
allennlp >= 0.8.2
tqdm >= 4.33.0
```

### Prepare pre-trained embedding:
#### BERT
Down the pytorch bert model, or convert tensorflow param yourself as follow:
```bash
export BERT_BASE_DIR=/users4/ythou/Projects/Resources/bert-base-uncased/uncased_L-12_H-768_A-12/

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch
  $BERT_BASE_DIR/bert_model.ckpt
  $BERT_BASE_DIR/bert_config.json
  $BERT_BASE_DIR/pytorch_model.bin
```
Set BERT path in the ./utils/config.py

### Prepare data
Get data at `./data/`

Full data is available by contacting me, or you can generate it by your self:

Set test, train, dev data file path in ./scripts/

#### Few-shot Data Generation Tool
We provide a generation tool for converting normal data into few-shot/meta-episode style. 
See details at [here](https://github.com/AtmaHou/MetaDialog#few-shot-data-construction-tool)


### Run!

Execute the command line to run with scripts:
```
source ./scripts/run_1_shot_slot_tagging.sh [gpu_id]
```

We provide all scripts for experiment at  `./scripts/`, and you can also directly run with `./main.py`.

bert based scripts:
- `run_b_stanford_1_main.sh`
- `run_b_stanford_5_main.sh`
- `run_b_toursg_1_main.sh`
- `run_b_toursg_5_main.sh`

electra based scripts:
- `run_e_stanford_1_main.sh`
- `run_e_stanford_5_main.sh`
- `run_e_toursg_1_main.sh`
- `run_e_toursg_5_main.sh`



