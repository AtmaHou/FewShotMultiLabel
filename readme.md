# Few-shot MLC

The code of paper [Few-Shot Learning for Multi-label Intent Detection](https://arxiv.org/abs/2010.05256).

The code framework is inherited [MetaDialog Framework](https://github.com/AtmaHou/MetaDialog), welcome to use it.

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
Original data is available by contacting me, or you can generate it:
Set test, train, dev data file path in ./scripts/

#### Few-shot Data Generation
We provide a generation tool for converting normal data into few-shot/meta-episode style. 


### The Scripts

All the scripts are saved in the folder `scripts`.
We only release the best setting scripts of our method.

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

#### [2020-12-28] add script to generate `tag_dict.all` file

- script: `scripts/get_tag_data_from_training_dataset.py`
- operation:
	- change the parameters called `MODEL_DIR` and `DATA_DIR`
- command: `python scripts/get_tag_data_from_training_dataset.py`
 


